use algorithmic_gas_benchmarks::convergence_cloning_empirical::validate_empirical;

#[test]
fn native_proposals_compare_independent_predictions_and_preserve_revival() {
    let report = futures_lite::future::block_on(validate_empirical(64, true)).unwrap();
    assert_eq!(report["summary"]["comparisons_violated"], 0);
    assert_eq!(report["summary"]["native_proposals"], 512);
    assert_eq!(report["summary"]["comparisons"], 34);
    let cases = report["cases"].as_array().unwrap();
    assert!(cases.iter().any(|c| c["alive"] == 1));
    for case in cases {
        assert_eq!(case["replicas"].as_array().unwrap().len(), 64);
        assert!(
            case["comparisons"]
                .as_array()
                .unwrap()
                .iter()
                .any(|c| c["id"] == "complete_two_swarm_weighted_drift")
        );
        assert!(
            case["comparisons"]
                .as_array()
                .unwrap()
                .iter()
                .all(|c| c["samples"] == 64 && c["not_rejected"] == true)
        );
    }
    assert!(
        report["balanced_keystone_cases"][0]["chi"]
            .as_f64()
            .unwrap()
            > 0.
    );
}

#[test]
fn insufficient_independent_sample_count_is_rejected() {
    assert!(futures_lite::future::block_on(validate_empirical(1, true)).is_err());
}
