use algorithmic_gas_benchmarks::{
    convergence_estimates::EstimateSuite, convergence_estimates_framework::validate_estimates,
};
use std::sync::OnceLock;

fn suite() -> &'static EstimateSuite {
    static SUITE: OnceLock<EstimateSuite> = OnceLock::new();
    SUITE.get_or_init(|| futures_lite::future::block_on(validate_estimates(128)).unwrap())
}

#[test]
fn individually_quoted_native_framework_estimates_pass() {
    let s = suite();
    assert_eq!(s.chapter, 1);
    assert!(s.evidence.len() > 300);
    for evidence in &s.evidence {
        assert!(
            evidence.passed(),
            "{} / {} / {:?}",
            evidence.source_labels.join(","),
            evidence.source_formula,
            evidence
                .checks
                .iter()
                .filter(|c| !c.passed)
                .collect::<Vec<_>>()
        );
        assert!(!evidence.source_formula.trim().is_empty());
    }
}

#[test]
fn native_nonidentical_measurement_mixture_is_exhaustive_and_normalizes_realized_values() {
    let cases: Vec<_> = suite()
        .evidence
        .iter()
        .filter(|e| {
            e.checks
                .iter()
                .any(|c| c.id == "native_nonidentical_mixture_raw")
        })
        .collect();
    assert_eq!(cases.len(), 3);
    for case in cases {
        assert!(
            case.inputs["N"]
                .as_u64()
                .is_some_and(|n| (2..=4).contains(&n))
        );
        assert!(
            case.checks
                .iter()
                .any(|c| c.id == "native_nonidentical_mixture_std")
        );
    }
    assert!(
        suite()
            .evidence
            .iter()
            .any(|e| e.inputs["outcomes"].as_u64() == Some(81))
    );
}

#[test]
fn antipodal_equal_fitness_control_has_zero_live_cloning() {
    let controls: Vec<_> = suite()
        .evidence
        .iter()
        .filter(|e| {
            e.checks
                .iter()
                .any(|c| c.id.starts_with("native_equal_fitness_zero_activity"))
        })
        .collect();
    assert!(
        controls
            .iter()
            .any(|e| e.inputs["positions"] == serde_json::json!([-0.8, 0.8]))
    );
    for e in controls {
        assert!(e.checks.iter().all(|c| c.observed == 0.));
    }
    assert!(suite().evidence.iter().any(|e| {
        e.checks
            .iter()
            .any(|c| c.id.starts_with("native_activity_gap"))
    }));
}

#[test]
fn normalized_status_and_structural_fraction_constants_are_population_uniform() {
    let cases: Vec<_> = suite()
        .evidence
        .iter()
        .filter(|e| {
            e.checks
                .iter()
                .any(|c| c.id == "normalized_status_N_uniform_modulus")
        })
        .collect();
    assert_eq!(cases.len(), 4);
    for e in cases {
        assert_eq!(e.inputs["single_position_modulus"], 0.25);
    }
    assert!(suite().evidence.iter().any(|e| {
        e.inputs["k2"] == 1
            && e.checks
                .iter()
                .any(|c| c.id == "native_refined_structural_fraction_bound")
    }));
    assert!(suite().evidence.iter().any(|e| {
        e.checks
            .iter()
            .any(|c| c.id.starts_with("uniform_exact_tv"))
    }));
}

#[test]
fn rejects_insufficient_samples() {
    assert!(futures_lite::future::block_on(validate_estimates(31)).is_err());
}
