//! Agreement with the reference estimators of `fragile.physics.geometry`.
//! Fixtures come from `tools/export_tessellation_fixtures.py`. Topology must
//! match exactly; numbers agree up to the conditioning of the different
//! eigen/SVD and linear solvers.
use algorithmic_gas::{
    ObservationBatch, TensorBatch,
    tessellation::{
        CurvatureKind, CurvatureSpec, GeometryPipelineConfig, Parallelism, TessellationGeometry,
        WeightMode, WeightSpec,
        forces::{GraphField, boris_rotate, curl, viscous_force},
    },
};
use serde_json::Value;

fn fixture(name: &str) -> Value {
    let path = format!(
        "{}/tests/fixtures/tessellation/{name}.json",
        env!("CARGO_MANIFEST_DIR")
    );
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}
fn flat(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(items) => items.iter().flat_map(flat).collect(),
        other => vec![other.as_f64().unwrap()],
    }
}
/// Directed edges in (source, destination) order, the CSR slot order.
fn edges(v: &Value) -> Vec<[u32; 2]> {
    let mut out: Vec<[u32; 2]> = v
        .as_array()
        .unwrap()
        .iter()
        .map(|e| [e[0].as_u64().unwrap() as u32, e[1].as_u64().unwrap() as u32])
        .collect();
    out.sort_unstable();
    out
}
#[track_caller]
fn close(label: &str, actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len(), "{label}: length");
    let scale = expected
        .iter()
        .fold(0f64, |m, v| m.max(v.abs()))
        .max(1e-300);
    for (k, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tolerance * (e.abs() + 1e-6 * scale),
            "{label}[{k}]: {a} vs {e}"
        );
    }
}
fn evaluate(d: usize, positions: Vec<f64>, length_scale: f64) -> TessellationGeometry<f64> {
    let n = positions.len() / d;
    let modes = [
        WeightMode::Uniform,
        WeightMode::InverseDistance,
        WeightMode::InverseVolume,
        WeightMode::InverseRiemannianVolume,
        WeightMode::InverseRiemannianDistance,
        WeightMode::Kernel,
        WeightMode::RiemannianKernel,
        WeightMode::RiemannianKernelVolume,
    ];
    let config = GeometryPipelineConfig {
        weights: modes
            .into_iter()
            .map(|mode| WeightSpec {
                length_scale,
                ..WeightSpec::new(mode)
            })
            .collect(),
        curvature: vec![
            CurvatureSpec {
                name: "proxy".into(),
                estimator: CurvatureKind::ConformalLaplacian {
                    weights: "inverse_riemannian_distance".into(),
                    det_floor: 1e-12,
                },
            },
            CurvatureSpec {
                name: "full".into(),
                estimator: CurvatureKind::ConformalQuadraticFit {
                    weights: Some("inverse_riemannian_distance".into()),
                    reg: 1e-6,
                    conformal_factor: true,
                    tensor: true,
                    det_floor: 1e-12,
                },
            },
        ],
        ..GeometryPipelineConfig::default()
    };
    let obs = ObservationBatch::positions(TensorBatch::vectors(n, d, positions).unwrap());
    config.validate::<f64>(d).unwrap();
    config
        .evaluate(&obs, &vec![true; n], None, 1 << 22)
        .unwrap()
}

#[test]
fn geometry_matches_the_reference_estimators() {
    for name in ["cloud_2d", "cloud_3d"] {
        let f = fixture(name);
        let d = f["dimension"].as_u64().unwrap() as usize;
        let g = evaluate(
            d,
            flat(&f["positions"]),
            f["length_scale"].as_f64().unwrap(),
        );
        assert_eq!(
            g.graph().coo(),
            edges(&f["edges"]),
            "{name}: Delaunay edges"
        );
        close("metric", &g.metric.metric, &flat(&f["metric"]), 1e-8);
        close(
            "metric_det",
            &g.metric.determinant,
            &flat(&f["metric_det"]),
            1e-8,
        );
        close("volume", &g.volume, &flat(&f["volume"]), 1e-8);
        close(
            "diffusion",
            &g.metric.diffusion,
            &flat(&f["diffusion"]),
            1e-8,
        );
        close(
            "edge_distances",
            &g.lengths.euclidean,
            &flat(&f["edge_distances"]),
            1e-12,
        );
        close(
            "edge_geodesic",
            &g.lengths.geodesic(),
            &flat(&f["edge_geodesic"]),
            1e-8,
        );
        for (mode, expected) in f["weights"].as_object().unwrap() {
            // The reference has unit cells, which makes its volume weights
            // uniform; here they use the Voronoi cell volumes.
            let key = if mode == "inverse_volume" {
                "uniform"
            } else {
                mode.as_str()
            };
            close(mode, &g.weights[key], &flat(expected), 1e-8);
        }
        close(
            "ricci_proxy",
            &g.curvature["proxy"].scalar,
            &flat(&f["ricci_proxy"]),
            1e-7,
        );
        // Normal equations with an absolute ridge: allow for their conditioning.
        close(
            "ricci_proxy_full",
            &g.curvature["full"].scalar,
            &flat(&f["ricci_proxy_full"]),
            1e-5,
        );
        close(
            "ricci_tensor_full",
            g.curvature["full"].tensor.as_ref().unwrap(),
            &flat(&f["ricci_tensor_full"]),
            1e-5,
        );

        let forces = &f["forces"];
        let positions = flat(&f["positions"]);
        let velocities = flat(&forces["velocities"]);
        let n = velocities.len() / d;
        let field = GraphField {
            graph: g.graph(),
            weights: &g.weights["riemannian_kernel_volume"],
            positions: &positions,
            dimension: d,
            eligible: &vec![true; n],
            wrap: &[],
        };
        let par = Parallelism::Serial;
        let nu = forces["nu"].as_f64().unwrap();
        let viscous = viscous_force(&field, &velocities, nu, par);
        close(
            "viscous_force",
            &viscous,
            &flat(&forces["viscous_force"]),
            1e-8,
        );
        let c = curl(&field, &viscous, par).unwrap();
        close("curl", &c, &flat(&forces["curl"]), 1e-6);
        // One B step: quarter kick, rotation over dt / 2, quarter kick.
        let duration = 0.5 * forces["dt"].as_f64().unwrap();
        let kicked: Vec<f64> = velocities
            .iter()
            .zip(&viscous)
            .map(|(v, f)| v + 0.5 * duration * f)
            .collect();
        let scale = 0.5 * forces["beta_curl"].as_f64().unwrap() * duration;
        let (rotated, angle) = boris_rotate(&kicked, &c, d, scale, &vec![true; n], par).unwrap();
        let second = viscous_force(&field, &rotated, nu, par);
        let result: Vec<f64> = rotated
            .iter()
            .zip(&second)
            .map(|(v, f)| v + 0.5 * duration * f)
            .collect();
        close(
            "kicked_velocities",
            &result,
            &flat(&forces["kicked_velocities"]),
            1e-8,
        );
        close(
            "rotation_angle",
            &angle,
            &flat(&forces["rotation_angle"]),
            1e-6,
        );
    }
}

#[test]
fn degenerate_swarms_match_the_reference_edge_rules() {
    for case in fixture("degenerate").as_array().unwrap() {
        let positions = case["positions"].as_array().unwrap();
        let d = positions[0].as_array().unwrap().len();
        let g = evaluate(d, flat(&case["positions"]), 1.);
        assert_eq!(g.graph().coo(), edges(&case["edges"]), "{}", case["name"]);
        assert!(g.curvature["proxy"].scalar.iter().all(|r| r.is_finite()));
    }
}
