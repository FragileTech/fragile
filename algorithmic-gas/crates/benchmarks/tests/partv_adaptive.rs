use algorithmic_gas::{
    Precision, TensorBatch,
    boundary::BoundaryPolicy,
    kinetic::KineticKind,
    partv_geometry::{
        FitnessInput, MetricPolicy, conditional_fitness_pipeline, metric_from_hessian,
    },
    tracking::RecordingConfig,
};
use algorithmic_gas_benchmarks::{Benchmark, RunConfig, physics_metric::PhysicsMetricConfig};
use futures_lite::future::block_on;
#[allow(clippy::field_reassign_with_default)]
fn config() -> RunConfig {
    let mut c = RunConfig::default();
    c.benchmark = Benchmark::Quadratic;
    c.walkers = 12;
    c.dimensions = 2;
    c.gas.precision = Precision::F64;
    c.gas.boundary = BoundaryPolicy::Unbounded;
    c.gas.kinetic.integrator = KineticKind::Baoab {
        positions: "positions".into(),
        velocities: "velocities".into(),
        dt: 0.18,
        friction: 0.7,
    };
    c.physics_metric = Some(PhysicsMetricConfig {
        epsilon: 0.2,
        temperature: 0.6,
        policy: MetricPolicy::Clipped,
        ..Default::default()
    });
    c
}
#[test]
fn adaptive_o_stage_uses_actual_query_and_frozen_sources() {
    block_on(async {
        let c = config();
        let mut gas = c.build::<f64>().await.unwrap();
        let mut p = gas.population().clone();
        p.observations.fields.insert(
            "velocities".into(),
            TensorBatch::vectors(
                c.walkers,
                2,
                (0..2 * c.walkers)
                    .map(|i| 0.5 + (i as f64) * 0.04)
                    .collect(),
            )
            .unwrap(),
        );
        gas.replace_population(p).await.unwrap();
        gas.start_recording(RecordingConfig::default()).unwrap();
        gas.step().await.unwrap();
        let s = &gas.recording().unwrap().steps[0];
        let actual = s.stages.iter().find(|s| s.stage == "A1").unwrap();
        let field = |name: &str| {
            s.field_evaluations
                .iter()
                .find(|f| f.stage == "O" && f.field == name)
                .unwrap()
        };
        let points: Vec<[f64; 2]> = s
            .before
            .observations
            .field("positions")
            .unwrap()
            .values()
            .chunks_exact(2)
            .map(|p| [p[0], p[1]])
            .collect();
        let velocities: Vec<[f64; 2]> = s
            .before
            .observations
            .field("velocities")
            .unwrap()
            .values()
            .chunks_exact(2)
            .map(|p| [p[0], p[1]])
            .collect();
        let companions: Vec<_> = s
            .report
            .distance_companions
            .indices
            .iter()
            .enumerate()
            .map(|(i, &j)| {
                if s.report.distance_companions.valid[i] {
                    s.report.distance_sources[j as usize].slot as usize
                } else {
                    i
                }
            })
            .collect();
        let mut differs_from_initial = 0.;
        for i in 0..c.walkers {
            let query = [
                actual.fields["positions"].values[2 * i],
                actual.fields["positions"].values[2 * i + 1],
            ];
            let mut input:FitnessInput=serde_json::from_value(serde_json::json!({"points":points,"velocities":velocities,"alive":s.report.pre_clone_eligible,"companions":companions,"companion_valid":s.report.distance_companions.valid,"target":i,"query":query,"objective":{"kind":"quadratic","curvature":[[1.,0.],[0.,1.]]},"metric_epsilon":0.2})).unwrap();
            let jet = conditional_fitness_pipeline(&input, &c.gas.fitness).unwrap();
            let metric = metric_from_hessian(jet.hessian, 0.2, MetricPolicy::Clipped).unwrap();
            for j in 0..4 {
                assert!(
                    (field("fitness_hessian").values[4 * i + j] - jet.hessian[j / 2][j % 2]).abs()
                        < 1e-11
                );
                assert!(
                    (field("fitness_metric").values[4 * i + j] - metric.metric[j / 2][j % 2]).abs()
                        < 1e-11
                );
            }
            input.query = points[i];
            let old = conditional_fitness_pipeline(&input, &c.gas.fitness).unwrap();
            differs_from_initial += jet
                .hessian
                .iter()
                .flatten()
                .zip(old.hessian.iter().flatten())
                .map(|(a, b)| (a - b).abs())
                .sum::<f64>();
        }
        assert!(differs_from_initial > 1e-3);
        assert!(
            s.field_evaluations
                .iter()
                .any(|f| f.stage == "B1" && f.field == "potential_gradient")
        );
        assert!(
            s.field_evaluations
                .iter()
                .any(|f| f.stage == "B2" && f.field == "potential_gradient")
        );
        let noise = s.noise.iter().find(|n| n.stage == "O").unwrap();
        assert_eq!(noise.factor.as_ref().unwrap().len(), c.walkers * 4);
    });
}
#[test]
fn recording_does_not_change_adaptive_rng_or_trajectory() {
    block_on(async {
        let c = config();
        let mut a = c.build::<f64>().await.unwrap();
        let mut b = c.build::<f64>().await.unwrap();
        a.start_recording(RecordingConfig::default()).unwrap();
        for _ in 0..3 {
            a.step().await.unwrap();
            b.step().await.unwrap();
        }
        assert_eq!(a.population(), b.population());
    });
}
#[test]
fn adaptive_configuration_rejects_nonsmooth_and_wrong_precision() {
    block_on(async {
        let mut c = config();
        assert!(c.build::<f32>().await.is_err());
        c.gas.fitness.reward_standardizer =
            algorithmic_gas::fitness::Standardizer::LegacySample { epsilon: 0.1 };
        assert!(c.build::<f64>().await.is_err());
    });
}
#[test]
fn revived_rows_record_donor_field_extension() {
    block_on(async {
        let c = config();
        let mut gas = c.build::<f64>().await.unwrap();
        let mut p = gas.population().clone();
        p.validity[0].terminated = true;
        gas.replace_population(p).await.unwrap();
        gas.start_recording(RecordingConfig::default()).unwrap();
        gas.step().await.unwrap();
        let s = &gas.recording().unwrap().steps[0];
        assert!(s.report.clone_plan.choices[0].revival);
        let extension = s
            .field_evaluations
            .iter()
            .find(|f| f.field == "revival_donor_field_extension")
            .unwrap();
        assert_eq!(extension.values[0], 1.);
        let target = s
            .field_evaluations
            .iter()
            .find(|f| f.field == "conditional_field_target_slot")
            .unwrap()
            .values[0] as usize;
        assert!(s.report.pre_clone_eligible[target]);
        assert_ne!(target, 0);
    });
}

#[test]
fn death_before_o_has_absent_metric_coverage_and_zero_factor() {
    block_on(async {
        let mut c = config();
        c.gas.fitness.reward_exponent = 0.;
        c.gas.fitness.diversity_exponent = 0.;
        c.gas.boundary = BoundaryPolicy::AbsorbingBox {
            field: "positions".into(),
            domain: algorithmic_gas::boundary::BoxDomain {
                lower: vec![-1.; 2],
                upper: vec![1.; 2],
            },
        };
        c.initial_lower = -0.2;
        c.initial_upper = 0.2;
        c.physics_metric.as_mut().unwrap().policy = MetricPolicy::Strict;
        let mut gas = c.build::<f64>().await.unwrap();
        let mut p = gas.population().clone();
        let mut v = vec![0.; 2 * c.walkers];
        v[0] = 100.;
        p.observations.fields.insert(
            "velocities".into(),
            TensorBatch::vectors(c.walkers, 2, v).unwrap(),
        );
        gas.replace_population(p).await.unwrap();
        gas.start_recording(RecordingConfig::default()).unwrap();
        gas.step().await.unwrap();
        let s = &gas.recording().unwrap().steps[0];
        let field = s
            .field_evaluations
            .iter()
            .find(|f| f.stage == "O" && f.field == "fitness_metric")
            .unwrap();
        assert!(!field.available[0]);
        assert!(field.available[1..].iter().all(|x| *x));
        let factors = s
            .noise
            .iter()
            .find(|n| n.stage == "O")
            .unwrap()
            .factor
            .as_ref()
            .unwrap();
        assert_eq!(&factors[..4], &[0.; 4]);
    });
}
