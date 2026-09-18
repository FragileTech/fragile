//! Analytic and structural checks of the tessellation geometry stage.
use algorithmic_gas::{
    ObservationBatch, TensorBatch,
    boundary::BoxDomain,
    tessellation::{
        CurvatureKind, CurvatureSpec, GeometryPipelineConfig, MetricKind, Parallelism,
        ReggeLengths, TessellationDomain, TessellationGeometry, VolumeKind, VoronoiCellsConfig,
        WeightMode, WeightSpec,
    },
};

/// Deterministic pseudo-random points in the unit cube. A hash, not a
/// Kronecker sequence: lattice sequences are full of collinear triples.
fn cloud(n: usize, d: usize) -> Vec<f64> {
    let mix = |mut z: u64| {
        z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    };
    (0..n * d)
        .map(|k| (mix(k as u64) >> 11) as f64 / (1u64 << 53) as f64)
        .collect()
}
fn observations(d: usize, x: Vec<f64>) -> ObservationBatch<f64> {
    ObservationBatch::positions(TensorBatch::vectors(x.len() / d, d, x).unwrap())
}
fn run(config: &GeometryPipelineConfig, obs: &ObservationBatch<f64>) -> TessellationGeometry<f64> {
    let n = obs.field("positions").unwrap().rows();
    config
        .validate::<f64>(obs.field("positions").unwrap().width())
        .unwrap();
    config.evaluate(obs, &vec![true; n], None, 1 << 24).unwrap()
}
fn unit_box(d: usize) -> BoxDomain {
    BoxDomain {
        lower: vec![0.; d],
        upper: vec![1.; d],
    }
}
fn spec(name: &str, estimator: CurvatureKind) -> CurvatureSpec {
    CurvatureSpec {
        name: name.into(),
        estimator,
    }
}

#[test]
fn flat_metric_has_zero_curvature_and_normalized_weights() {
    let config = GeometryPipelineConfig {
        metric: MetricKind::Identity,
        weights: [
            WeightMode::Uniform,
            WeightMode::InverseDistance,
            WeightMode::InverseRiemannianDistance,
            WeightMode::Kernel,
            WeightMode::RiemannianKernel,
            WeightMode::RiemannianKernelVolume,
            WeightMode::InverseVolume,
            WeightMode::InverseRiemannianVolume,
        ]
        .map(WeightSpec::new)
        .to_vec(),
        curvature: vec![
            spec(
                "laplacian",
                CurvatureKind::ConformalLaplacian {
                    weights: "inverse_riemannian_distance".into(),
                    det_floor: 1e-12,
                },
            ),
            spec(
                "fit",
                CurvatureKind::ConformalQuadraticFit {
                    weights: None,
                    reg: 1e-6,
                    conformal_factor: true,
                    tensor: true,
                    det_floor: 1e-12,
                },
            ),
        ],
        ..GeometryPipelineConfig::default()
    };
    let g = run(&config, &observations(2, cloud(120, 2)));
    assert!(g.curvature["laplacian"].scalar.iter().all(|&r| r == 0.));
    assert!(g.curvature["fit"].scalar.iter().all(|&r| r.abs() < 1e-12));
    assert!(g.volume.iter().all(|&v| v == 1.));
    let graph = g.graph();
    graph.validate().unwrap();
    for w in g.weights.values() {
        for i in 0..graph.nodes() {
            let sum: f64 = w[graph.range(i)].iter().sum();
            assert!((sum - 1.).abs() < 1e-12);
        }
    }
    // With the identity metric the Riemannian kernel equals the Euclidean one
    // and geodesic lengths equal coordinate lengths.
    for (a, b) in g.weights["kernel"]
        .iter()
        .zip(&g.weights["riemannian_kernel"])
    {
        assert!((a - b).abs() < 1e-14);
    }
    for (a, b) in g.lengths.euclidean.iter().zip(g.lengths.geodesic()) {
        assert!((a - b).abs() < 1e-14);
    }
}

#[test]
fn quadratic_fit_recovers_conformally_flat_curvature() {
    for d in [2usize, 3] {
        let n = if d == 2 { 400 } else { 700 };
        let x = cloud(n, d);
        // u = a.x + x^T B x / 2 with a fixed symmetric B.
        let a = [0.3, -0.2, 0.1];
        let b = [[0.8, 0.1, -0.2], [0.1, -0.5, 0.05], [-0.2, 0.05, 0.4]];
        let potential = |p: &[f64]| -> f64 {
            (0..d).map(|i| a[i] * p[i]).sum::<f64>()
                + 0.5
                    * (0..d)
                        .flat_map(|i| (0..d).map(move |j| (i, j)))
                        .map(|(i, j)| b[i][j] * p[i] * p[j])
                        .sum::<f64>()
        };
        let mut metric = Vec::with_capacity(n * d * d);
        for p in x.chunks_exact(d) {
            let scale = (2. * potential(p)).exp();
            for i in 0..d {
                for j in 0..d {
                    metric.push(if i == j { scale } else { 0. });
                }
            }
        }
        let mut obs = observations(d, x.clone());
        obs.fields
            .insert("g".into(), TensorBatch::new(n, vec![d, d], metric).unwrap());
        let config = GeometryPipelineConfig {
            metric: MetricKind::ObservationField {
                field: "g".into(),
                min_eig: None,
                max_eig: None,
            },
            curvature: vec![spec(
                "fit",
                CurvatureKind::ConformalQuadraticFit {
                    weights: None,
                    // The ridge is absolute: keep it far below the squared spacing.
                    reg: 1e-18,
                    conformal_factor: true,
                    tensor: true,
                    det_floor: 1e-300,
                },
            )],
            ..GeometryPipelineConfig::default()
        };
        let g = run(&config, &obs);
        let out = &g.curvature["fit"];
        let df = d as f64;
        let laplacian: f64 = (0..d).map(|i| b[i][i]).sum();
        let mut checked = 0;
        for (i, p) in x.chunks_exact(d).enumerate() {
            if g.tessellation.mesh.hull[g.tessellation.sites.site_of_walker[i] as usize]
                || g.graph().degree(i) < d + d * (d + 1) / 2
            {
                continue;
            }
            let grad: Vec<f64> = (0..d)
                .map(|r| a[r] + (0..d).map(|c| b[r][c] * p[c]).sum::<f64>())
                .collect();
            let norm: f64 = grad.iter().map(|v| v * v).sum();
            let exact =
                -2. * (df - 1.) * (-2. * potential(p)).exp() * (laplacian + 0.5 * (df - 2.) * norm);
            assert!(
                (out.scalar[i] - exact).abs() < 1e-5 * (1. + exact.abs()),
                "d={d} i={i}: {} vs {exact}",
                out.scalar[i]
            );
            // Trace of the coordinate Ricci tensor with g^{-1} = e^{-2u} delta.
            let t = out.tensor.as_ref().unwrap();
            let trace: f64 = (0..d).map(|k| t[i * d * d + k * d + k]).sum();
            assert!(((-2. * potential(p)).exp() * trace - exact).abs() < 1e-5 * (1. + exact.abs()));
            checked += 1;
        }
        assert!(checked > n / 4, "d={d}: only {checked} interior walkers");
    }
}

#[test]
fn voronoi_cells_close_the_domain() {
    for d in [2usize, 3] {
        let n = if d == 2 { 90 } else { 70 };
        for periodic in [true, false] {
            let domain = if periodic {
                TessellationDomain::Periodic {
                    bounds: unit_box(d),
                }
            } else {
                TessellationDomain::ClipBox {
                    bounds: unit_box(d),
                }
            };
            let config = GeometryPipelineConfig {
                domain,
                volume: VolumeKind::VoronoiCell,
                cells: Some(VoronoiCellsConfig::default()),
                weights: vec![WeightSpec {
                    normalize: false,
                    ..WeightSpec::new(WeightMode::FacetArea)
                }],
                curvature: vec![],
                ..GeometryPipelineConfig::default()
            };
            let g = run(&config, &observations(d, cloud(n, d)));
            let cells = g.cells.as_ref().unwrap();
            assert!(
                cells.bounded.iter().all(|&b| b),
                "d={d} periodic={periodic}"
            );
            let total: f64 = cells.volume.iter().sum();
            assert!(
                (total - 1.).abs() < 1e-10,
                "d={d} periodic={periodic}: {total}"
            );
            let graph = g.graph();
            let area = &g.weights["facet_area"];
            for e in 0..graph.edges() {
                assert!((area[e] - area[graph.reverse()[e] as usize]).abs() < 1e-12);
                assert!(area[e] >= 0.);
            }
            if periodic {
                assert!(cells.tier.iter().all(|&t| t == 2));
                // Minimum-image edges never exceed half the box diagonal.
                let limit = 0.5 * (d as f64).sqrt();
                assert!(g.lengths.euclidean.iter().all(|&l| l <= limit));
            } else {
                // Cells cut by the box are boundary cells; a planar swarm of
                // this size also has interior ones.
                assert!(cells.tier.contains(&0));
                assert!(d == 3 || cells.tier.contains(&2));
            }
        }
    }
}

#[test]
fn open_domain_marks_hull_cells_unbounded() {
    let config = GeometryPipelineConfig {
        cells: Some(VoronoiCellsConfig::default()),
        curvature: vec![
            spec("volume", CurvatureKind::VolumeDistortion),
            spec("shape", CurvatureKind::ShapeDistortion),
        ],
        ..GeometryPipelineConfig::default()
    };
    let g = run(&config, &observations(2, cloud(80, 2)));
    let cells = g.cells.as_ref().unwrap();
    let sites = &g.tessellation.sites;
    for i in 0..80 {
        let hull = g.tessellation.mesh.hull[sites.site_of_walker[i] as usize];
        assert_eq!(cells.bounded[i], !hull);
        assert_eq!(g.curvature["volume"].valid[i], !hull);
        if !hull {
            assert!(cells.volume[i] > 0.);
            let s = g.curvature["shape"].scalar[i];
            assert!((0. ..1.).contains(&s));
        }
    }
    // The valid volume distortions average to zero by construction.
    assert!(g.curvature["volume"].total.unwrap().abs() < 1e-12);
}

#[test]
fn regge_deficits_vanish_in_flat_space_and_integrate_curvature() {
    for d in [2usize, 3] {
        let config = GeometryPipelineConfig {
            metric: MetricKind::Identity,
            curvature: vec![spec(
                "regge",
                CurvatureKind::ReggeDeficit {
                    lengths: ReggeLengths::Euclidean,
                },
            )],
            ..GeometryPipelineConfig::default()
        };
        let g = run(
            &config,
            &observations(d, cloud(if d == 2 { 100 } else { 60 }, d)),
        );
        let out = &g.curvature["regge"];
        assert!(out.valid.iter().any(|&v| v));
        // Angles from lengths are ill-conditioned at the obtuse corner of a
        // sliver next to the hull, so flat space is recovered to ~sqrt(eps).
        let worst = out.scalar.iter().fold(0f64, |m, r| m.max(r.abs()));
        assert!(worst < 1e-4, "d={d}: worst flat Regge curvature {worst}");
        assert!(out.total.unwrap().abs() < 1e-6);
    }
    // Unit sphere in stereographic coordinates, R = 2. With exact great-circle
    // edge lengths the Regge action per area converges to 2.
    let n = 1500;
    let x: Vec<f64> = cloud(n, 2).into_iter().map(|v| 0.6 * (v - 0.5)).collect();
    let config = GeometryPipelineConfig {
        metric: MetricKind::Identity,
        curvature: vec![],
        ..GeometryPipelineConfig::default()
    };
    let g = run(&config, &observations(2, x.clone()));
    let lift = |i: u32| {
        let (u, v) = (x[2 * i as usize], x[2 * i as usize + 1]);
        let s = 1. + u * u + v * v;
        [2. * u / s, 2. * v / s, (s - 2.) / s]
    };
    let exact: Vec<f64> = g
        .graph()
        .coo()
        .into_iter()
        .map(|[i, j]| {
            let (p, q) = (lift(i), lift(j));
            let chord = (0..3).map(|k| (p[k] - q[k]).powi(2)).sum::<f64>().sqrt();
            2. * (0.5 * chord).asin()
        })
        .collect();
    let r = algorithmic_gas::tessellation::regge::curvature(
        &g.tessellation,
        &exact,
        Parallelism::Serial,
    );
    let area: f64 = r
        .dual_volume
        .iter()
        .zip(&r.valid)
        .filter(|p| *p.1)
        .map(|p| *p.0)
        .sum();
    assert!(area > 1., "valid Regge area {area}");
    assert!(
        (r.action / area - 2.).abs() < 0.02,
        "Regge action per area {} on area {area}",
        r.action / area
    );
}

#[test]
fn coincident_and_degenerate_swarms_are_handled_without_perturbation() {
    let config = GeometryPipelineConfig::default();
    // Every walker at the origin: complete graph, flat, g = 1e5 I.
    let g = run(&config, &observations(2, vec![0.; 24]));
    assert_eq!(g.graph().edges(), 12 * 11);
    assert!(g.curvature["ricci_scalar"].scalar.iter().all(|&r| r == 0.));
    assert!(g.volume.iter().all(|&v| (v - 1e5).abs() < 1e-4));
    // Collinear walkers in 3D form a path.
    let line: Vec<f64> = (0..6)
        .flat_map(|i| [i as f64, 2. * i as f64, 0.5])
        .collect();
    let g = run(&config, &observations(3, line));
    assert_eq!((g.tessellation.rank, g.graph().edges()), (1, 10));
    assert!(
        g.curvature["ricci_scalar"]
            .scalar
            .iter()
            .all(|r| r.is_finite())
    );
}

#[test]
fn serial_and_parallel_paths_are_bit_identical() {
    for d in [2usize, 3] {
        let base = GeometryPipelineConfig {
            domain: TessellationDomain::Periodic {
                bounds: unit_box(d),
            },
            cells: Some(VoronoiCellsConfig::default()),
            weights: [
                WeightMode::InverseRiemannianDistance,
                WeightMode::RiemannianKernelVolume,
                WeightMode::FacetAreaOverDistance,
            ]
            .map(WeightSpec::new)
            .to_vec(),
            curvature: vec![
                spec(
                    "laplacian",
                    CurvatureKind::ConformalLaplacian {
                        weights: "inverse_riemannian_distance".into(),
                        det_floor: 1e-12,
                    },
                ),
                spec(
                    "fit",
                    CurvatureKind::ConformalQuadraticFit {
                        weights: Some("riemannian_kernel_volume".into()),
                        reg: 1e-6,
                        conformal_factor: true,
                        tensor: true,
                        det_floor: 1e-12,
                    },
                ),
                spec(
                    "regge",
                    CurvatureKind::ReggeDeficit {
                        lengths: ReggeLengths::Geodesic,
                    },
                ),
                spec("shape", CurvatureKind::ShapeDistortion),
            ],
            ..GeometryPipelineConfig::default()
        };
        let obs = observations(d, cloud(300, d));
        let serial = run(
            &GeometryPipelineConfig {
                parallelism: Parallelism::Serial,
                ..base.clone()
            },
            &obs,
        );
        let parallel = run(
            &GeometryPipelineConfig {
                parallelism: Parallelism::Always,
                ..base
            },
            &obs,
        );
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(serial.tessellation, parallel.tessellation);
        assert_eq!(bits(&serial.metric.metric), bits(&parallel.metric.metric));
        assert_eq!(
            bits(&serial.metric.diffusion),
            bits(&parallel.metric.diffusion)
        );
        assert_eq!(bits(&serial.volume), bits(&parallel.volume));
        assert_eq!(
            bits(&serial.lengths.geodesic_sq),
            bits(&parallel.lengths.geodesic_sq)
        );
        for (k, w) in &serial.weights {
            assert_eq!(bits(w), bits(&parallel.weights[k]), "{k}");
        }
        for (k, c) in &serial.curvature {
            assert_eq!(bits(&c.scalar), bits(&parallel.curvature[k].scalar), "{k}");
        }
        let (a, b) = (serial.cells.unwrap(), parallel.cells.unwrap());
        assert_eq!(bits(&a.volume), bits(&b.volume));
        assert_eq!(bits(&a.facet_area), bits(&b.facet_area));
        assert_eq!(bits(&a.vertices), bits(&b.vertices));
    }
}

#[test]
fn configuration_round_trips_and_rejects_dangling_references() {
    let config = GeometryPipelineConfig::default();
    let json = serde_json::to_string(&config).unwrap();
    assert_eq!(
        serde_json::from_str::<GeometryPipelineConfig>(&json).unwrap(),
        config
    );
    let dangling = GeometryPipelineConfig {
        weights: vec![WeightSpec::new(WeightMode::Kernel)],
        ..GeometryPipelineConfig::default()
    };
    assert!(dangling.validate::<f64>(2).is_err());
    // Four projected coordinates have no tessellator.
    assert!(config.validate::<f64>(4).is_err());
    assert!(serde_json::from_str::<GeometryPipelineConfig>(r#"{"unknown":1}"#).is_err());
}
