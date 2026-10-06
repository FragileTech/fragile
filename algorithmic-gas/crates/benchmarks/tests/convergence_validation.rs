use algorithmic_gas_benchmarks::convergence_validation::{
    ValidationConfig, coverage, inventory, run_validation,
};
use serde_json::json;
use std::collections::BTreeSet;

#[test]
fn inventory_covers_each_source_label_and_retains_hypotheses() {
    let docs = [
        include_str!(
            "../../../docs/source/2_fractal_gas/convergence_program/01_fragile_gas_framework.md"
        ),
        include_str!(
            "../../../docs/source/2_fractal_gas/convergence_program/02_euclidean_gas.md"
        ),
        include_str!("../../../docs/source/2_fractal_gas/convergence_program/03_cloning.md"),
    ];
    for (chapter, source) in inventory().unwrap().iter().zip(docs) {
        let inventoried: BTreeSet<_> = chapter["formal_items"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|item| item["label"].as_str())
            .collect();
        for line in source.lines() {
            if let Some(label) = line.trim().strip_prefix(":label:") {
                // Remarks are tracked as formal context, with individual claims
                // retained by the generator's required-item registry.
                let label = label.trim();
                if label.starts_with("rem")
                    || label.starts_with("remark")
                    || label.starts_with("proof")
                {
                    continue;
                }
                assert!(inventoried.contains(label), "missing {label}");
            }
        }
        assert!(
            !chapter["quantitative_expressions"]
                .as_array()
                .unwrap()
                .is_empty()
        );
    }
}

#[test]
fn inventory_entries_do_not_automatically_count_as_validated() {
    let catalog = vec![json!({"formal_items":[
        {"label":"observed","computability_kind":"empirical"},
        {"label":"uncomputed","computability_kind":"exact"},
        {"label":"topology","computability_kind":"not_numerically_testable"}
    ]})];
    let report = coverage(&catalog, &BTreeSet::from(["observed".into()]));
    assert_eq!(
        report["formal_claim_counts"]["diagnostic_exercised_under_stated_scope"],
        1
    );
    assert_eq!(
        report["formal_claim_counts"]["not_yet_numerically_validated"],
        1
    );
    assert_eq!(report["formal_claim_counts"]["analytical_only"], 1);
}

#[test]
fn validation_matrix_is_explicit_reproducible_and_rejects_unknown_profiles() {
    let config = ValidationConfig::default();
    assert_eq!(config.matrix().unwrap().len(), 96);
    assert_eq!(config.matrix().unwrap(), config.matrix().unwrap());
    let invalid = ValidationConfig {
        profiles: vec!["typo".into()],
        ..config.clone()
    };
    assert!(invalid.validate().is_err());
    let invalid = ValidationConfig {
        dimensions: vec![0],
        ..config
    };
    assert!(invalid.validate().is_err());
}

#[test]
fn native_smoke_report_counts_exact_and_statistical_checks_separately() {
    let config = ValidationConfig {
        walkers: vec![4],
        dimensions: vec![1],
        seeds: vec![7],
        landscapes: vec![algorithmic_gas_benchmarks::Benchmark::Quadratic],
        profiles: vec!["canonical".into()],
        steps: 2,
        kinetic_samples: 256,
        cloning_samples: 256,
        decay: None,
        shared_decay: None,
        runs: vec![],
    };
    let report = futures_lite::future::block_on(run_validation(&config)).unwrap();
    assert_eq!(report["summary"]["runs"], 1);
    assert_eq!(report["summary"]["exact_checks_failed"], 0);
    assert_eq!(report["summary"]["run_errors"], 0);
    assert_eq!(report["runs"][0]["completed_steps"], 2);
    assert!(
        report["summary"]["kinetic_statistical_checks_not_rejected"]
            .as_u64()
            .unwrap()
            > 0
    );
    assert!(
        !report["canonical_keystone_constants"]
            .as_array()
            .unwrap()
            .is_empty()
    );
}

#[test]
fn paired_complete_update_diagnostics_are_integrated_without_counting_initial_points() {
    use algorithmic_gas_benchmarks::convergence_decay::{
        DecayProfile, DecayValidationConfig, PairRandomness,
    };
    let decay = DecayValidationConfig {
        walkers: vec![4],
        dimensions: vec![1],
        seeds: vec![7, 1729],
        steps: 4,
        fit_end_step: 4,
        profiles: vec![DecayProfile::LinearQuadraticNoiseless],
        ..DecayValidationConfig::default()
    };
    let config = ValidationConfig {
        walkers: vec![4],
        dimensions: vec![1],
        seeds: vec![7],
        landscapes: vec![algorithmic_gas_benchmarks::Benchmark::Quadratic],
        profiles: vec!["canonical".into()],
        steps: 1,
        kinetic_samples: 64,
        cloning_samples: 64,
        runs: vec![],
        shared_decay: Some(DecayValidationConfig {
            pair_randomness: PairRandomness::Shared,
            ..decay.clone()
        }),
        decay: Some(decay),
    };
    let report = futures_lite::future::block_on(run_validation(&config)).unwrap();
    assert_eq!(report["summary"]["phase_errors"], 0);
    assert_eq!(report["summary"]["exact_checks_failed"], 0);
    assert_eq!(report["summary"]["paired_decay_cases"], 2);
    assert_eq!(report["summary"]["paired_decay_trajectories"], 4);
    assert_eq!(report["summary"]["paired_complete_update_observations"], 16);
    assert_eq!(
        report["complete_update_decay"]["cases"][0]["ensemble_trajectory"]
            .as_array()
            .unwrap()
            .len(),
        5
    );
    assert_eq!(
        report["complete_update_decay"]["cases"][0]["endpoint_decline"]["empirical_mean_decreased"],
        true
    );
    assert_eq!(
        report["shared_update_decay"]["cases"][0]["endpoint_decline"]["empirical_mean_decreased"],
        true
    );
}

#[test]
fn valid_custom_singleton_runs_without_an_auxiliary_k2_fixture() {
    let config = ValidationConfig {
        runs: vec![algorithmic_gas_benchmarks::RunConfig {
            walkers: 1,
            dimensions: 1,
            gas: algorithmic_gas::GasConfig::euclidean(1, 0.04).unwrap(),
            ..algorithmic_gas_benchmarks::RunConfig::euclidean().unwrap()
        }],
        steps: 1,
        kinetic_samples: 64,
        cloning_samples: 64,
        decay: None,
        shared_decay: None,
        ..ValidationConfig::default()
    };
    let report = futures_lite::future::block_on(run_validation(&config)).unwrap();
    assert_eq!(report["summary"]["run_errors"], 0);
    assert_eq!(report["runs"][0]["completed_steps"], 1);
    assert_eq!(
        report["auxiliary_fixture_unavailable"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
}

#[test]
fn auxiliary_overflow_preserves_actual_engine_evidence() {
    let mut run = algorithmic_gas_benchmarks::RunConfig::euclidean().unwrap();
    run.walkers = 4;
    run.initial_lower = 0.;
    run.initial_upper = 0.;
    run.gas.fitness.reward_standardizer =
        algorithmic_gas::fitness::Standardizer::Global { sigma_min: 1e-100 };
    let config = ValidationConfig {
        runs: vec![run],
        decay: None,
        shared_decay: None,
        steps: 1,
        kinetic_samples: 64,
        cloning_samples: 64,
        ..ValidationConfig::default()
    };
    let report = futures_lite::future::block_on(run_validation(&config)).unwrap();
    assert_eq!(report["summary"]["phase_errors"], 1);
    assert_eq!(report["runs"][0]["completed_steps"], 1);
    assert_eq!(report["runs"][0]["trajectory"].as_array().unwrap().len(), 2);
}
