//! Agreement with the reference estimators of `fragile.physics.geometry`.
//! Fixtures come from `tools/export_tessellation_fixtures.py`. Topology must
//! match exactly; numbers agree up to the conditioning of the different
//! eigen/SVD and linear solvers. The reference regularizes the displacement
//! covariance with an absolute `1e-5 I`, so the parity run selects
//! `RidgeScale::Absolute`: that convention, not the scale-covariant default,
//! is what these fixtures were measured with. The last test pins the size of
//! the departure the shipped default makes from them, and why it is made.
use algorithmic_gas::{
    ObservationBatch, TensorBatch,
    partv_geometry::MetricPolicy,
    tessellation::{
        CurvatureKind, CurvatureSpec, GeometryPipelineConfig, MetricKind, Parallelism, RidgeScale,
        TessellationGeometry, WeightMode, WeightSpec,
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
/// Relative deviation of every entry, on the scale the tolerances are quoted
/// in: the entry's own magnitude plus `1e-6` of the largest expected one, so a
/// number that is zero in the reference is measured against the vector.
fn deviations(actual: &[f64], expected: &[f64]) -> Vec<f64> {
    let scale = expected
        .iter()
        .fold(0f64, |m, v| m.max(v.abs()))
        .max(1e-300);
    actual
        .iter()
        .zip(expected)
        .map(|(a, e)| (a - e).abs() / (e.abs() + 1e-6 * scale))
        .collect()
}
#[track_caller]
fn close(label: &str, actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len(), "{label}: length");
    for (k, deviation) in deviations(actual, expected).iter().enumerate() {
        assert!(
            *deviation <= tolerance,
            "{label}[{k}]: {} vs {}",
            actual[k],
            expected[k]
        );
    }
}
/// The largest entry of `deviations`: how far a whole vector has moved.
#[track_caller]
fn departure(label: &str, actual: &[f64], expected: &[f64]) -> f64 {
    assert_eq!(actual.len(), expected.len(), "{label}: length");
    deviations(actual, expected)
        .into_iter()
        .fold(0f64, f64::max)
}
/// The measured departure, to the digits recorded in the test that calls this.
#[track_caller]
fn pins(label: &str, measured: f64, recorded: f64) {
    assert!(
        (measured - recorded).abs() <= 1e-3 * recorded,
        "{label}: departure {measured:e}, recorded {recorded:e}"
    );
}
fn evaluate(d: usize, positions: Vec<f64>, length_scale: f64) -> TessellationGeometry<f64> {
    evaluate_with(d, positions, length_scale, RidgeScale::Absolute)
}
fn evaluate_with(
    d: usize,
    positions: Vec<f64>,
    length_scale: f64,
    scale: RidgeScale,
) -> TessellationGeometry<f64> {
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
        metric: MetricKind::NeighborCovariance {
            ridge: 1e-5,
            min_eig: Some(1e-6),
            max_eig: None,
            scale,
            policy: MetricPolicy::default(),
        },
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

/// The shipped default makes the ridge a multiple of the covariance trace, so
/// it cannot reproduce fixtures measured with an absolute `1e-5 I` — and it
/// must not: that is the correction the audit records as Q30. The test above
/// pins agreement with the reference under the reference's own convention;
/// this one pins how far the default stands from it, and that nothing the
/// metric does not enter moves with it. A default silently restored to the
/// absolute ridge fails here instead of passing parity.
#[test]
fn the_default_ridge_scale_departs_from_the_reference_metric() {
    // Largest relative departure of the default from the fixture, measured.
    for (name, metric_gap, det_gap) in [
        ("cloud_2d", 6.681938e-2, 6.681975e-2),
        ("cloud_3d", 4.190650e-3, 2.081267e-3),
    ] {
        let f = fixture(name);
        let d = f["dimension"].as_u64().unwrap() as usize;
        let length_scale = f["length_scale"].as_f64().unwrap();
        let positions = flat(&f["positions"]);
        let n = positions.len() / d;
        let reference = evaluate(d, positions.clone(), length_scale);
        let shipped = evaluate_with(
            d,
            positions.clone(),
            length_scale,
            RidgeScale::RelativeToTrace,
        );
        assert_eq!(RidgeScale::default(), RidgeScale::RelativeToTrace);
        // A ridge is not a site: the tessellation and every length and weight
        // built from the coordinates alone are bit identical to the parity run.
        assert_eq!(
            shipped.graph().coo(),
            edges(&f["edges"]),
            "{name}: Delaunay edges"
        );
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(
            bits(&shipped.lengths.euclidean),
            bits(&reference.lengths.euclidean),
            "{name}: edge distances"
        );
        for mode in ["uniform", "inverse_distance", "inverse_volume", "kernel"] {
            assert_eq!(
                bits(&shipped.weights[mode]),
                bits(&reference.weights[mode]),
                "{name}: {mode} weights"
            );
        }
        // The metric and what is built on it do move, by this much.
        pins(
            &format!("{name}: metric"),
            departure("metric", &shipped.metric.metric, &flat(&f["metric"])),
            metric_gap,
        );
        pins(
            &format!("{name}: metric_det"),
            departure(
                "metric_det",
                &shipped.metric.determinant,
                &flat(&f["metric_det"]),
            ),
            det_gap,
        );
        // Under x -> lambda x the displacement covariance scales as lambda^2,
        // so det(g) must scale as lambda^{-2d}. The reference ridge does not
        // scale, and at lambda = 1e-3 it dominates: every walker's determinant
        // is short by more than 1e3, the worst by 5.6e6 in 2D and 2.0e8 in 3D.
        // The default holds the exponent to roundoff.
        let lambda = 1e-3;
        let shrunk: Vec<f64> = positions.iter().map(|v| lambda * v).collect();
        let factor = lambda.powi(2 * d as i32);
        let small_reference = evaluate(d, shrunk.clone(), length_scale);
        let small_shipped = evaluate_with(d, shrunk, length_scale, RidgeScale::RelativeToTrace);
        let mut shortfall = 0f64;
        for i in 0..n {
            let stale =
                factor * small_reference.metric.determinant[i] / reference.metric.determinant[i];
            assert!(
                stale < 1e-3,
                "{name}[{i}]: the absolute ridge kept det g to {stale} of its covariant value"
            );
            shortfall = shortfall.max(1. / stale);
            let covariant =
                factor * small_shipped.metric.determinant[i] / shipped.metric.determinant[i];
            assert!(
                (covariant - 1.).abs() <= 1e-12,
                "{name}[{i}]: det g(lambda x) / lambda^-2d det g(x) = {covariant}"
            );
        }
        assert!(
            shortfall > 1e6,
            "{name}: worst absolute-ridge shortfall {shortfall}"
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
