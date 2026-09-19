//! Reference run configurations of the gas variant registry.
use algorithmic_gas::{GasError, RecordingConfig, variants::Variant};
use algorithmic_gas_benchmarks::{Benchmark, RunConfig, lecture_meanfield};
use futures_lite::future::block_on;

#[test]
fn every_implemented_variant_has_a_valid_reference_run_configuration() {
    for &variant in Variant::all().iter().filter(|v| v.implemented()) {
        let config = RunConfig::variant(variant.name()).unwrap();
        config.validate().unwrap();
        let reference = variant.reference().unwrap();
        assert_eq!(config.walkers, reference.walkers, "{}", variant.name());
        assert_eq!(config.dimensions, reference.dimensions);
        assert_eq!(config.gas, variant.default_config().unwrap());
        let json = serde_json::to_string(&config).unwrap();
        assert_eq!(serde_json::from_str::<RunConfig>(&json).unwrap(), config);
    }
    assert_eq!(
        RunConfig::variant("einstein-hilbert").unwrap(),
        RunConfig::einstein_hilbert().unwrap()
    );
}

#[test]
fn book_only_and_unknown_variants_are_distinct_errors() {
    for &variant in Variant::all().iter().filter(|v| !v.implemented()) {
        assert!(matches!(
            RunConfig::variant(variant.name()),
            Err(GasError::Capability(_))
        ));
    }
    assert!(matches!(
        RunConfig::variant("algorithmic"),
        Err(GasError::Configuration(_))
    ));
}

#[test]
fn euclidean_reference_is_the_canonical_lecture_configuration() {
    let config = RunConfig::euclidean().unwrap();
    assert_eq!(config.benchmark, Benchmark::Quadratic);
    let lecture = lecture_meanfield::config("III-03", &serde_json::json!({}), config.gas.seed);
    assert_eq!(config, lecture.unwrap());
}

/// Largest recorded viscous-force component and largest component of the
/// force summed over walkers, over a few recorded steps.
fn viscous_force(mut config: RunConfig) -> (f64, f64) {
    config.walkers = 24;
    let dimensions = config.dimensions;
    let mut gas = block_on(config.build::<f64>()).unwrap();
    gas.start_recording(RecordingConfig::default()).unwrap();
    for _ in 0..3 {
        block_on(gas.step()).unwrap();
    }
    let archive = gas.stop_recording().unwrap();
    let (mut largest, mut imbalance) = (0f64, 0f64);
    let mut records = 0;
    for step in &archive.steps {
        for record in step
            .field_evaluations
            .iter()
            .filter(|r| r.field == "viscous_force")
        {
            records += 1;
            assert!(record.available.iter().all(|&a| a));
            let mut total = vec![0f64; dimensions];
            for row in record.values.chunks(dimensions) {
                for (sum, &component) in total.iter_mut().zip(row) {
                    *sum += component;
                    largest = largest.max(component.abs());
                }
            }
            imbalance = total.iter().fold(imbalance, |m, x| m.max(x.abs()));
        }
    }
    // Both B kicks of every step.
    assert_eq!(records, 6);
    (largest, imbalance)
}

#[test]
fn viscous_euclidean_records_a_momentum_conserving_force_and_euclidean_a_zero_one() {
    let (largest, imbalance) = viscous_force(RunConfig::viscous_euclidean().unwrap());
    assert!(largest > 1e-6, "largest viscous force {largest}");
    // Eligible-count normalization: the pairwise terms cancel in the sum.
    assert!(imbalance < 1e-12 * largest.max(1.), "imbalance {imbalance}");

    let (largest, _) = viscous_force(RunConfig::euclidean().unwrap());
    assert_eq!(largest, 0.);
}
