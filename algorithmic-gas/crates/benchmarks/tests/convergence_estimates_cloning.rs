use algorithmic_gas_benchmarks::convergence_estimates_cloning::validate_estimates;

#[test]
fn native_laws_and_directed_certificates_satisfy_bound_checks() {
    let suite =
        futures_lite::future::block_on(validate_estimates(16)).expect("chapter 3 estimate suite");
    assert_eq!(suite.chapter, 3);
    assert!(suite.evidence.len() >= 50);
    let failed: Vec<_> = suite
        .evidence
        .iter()
        .filter(|e| !e.passed())
        .map(|e| {
            (
                &e.source_labels,
                &e.source_formula,
                e.hypothesis_checks
                    .iter()
                    .chain(&e.checks)
                    .filter(|c| !c.passed)
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    assert!(failed.is_empty(), "failed evidence: {failed:#?}");
    let source =
        include_str!("../../../docs/source/2_fractal_gas/convergence_program/03_cloning.md");
    for item in &suite.evidence {
        assert!(
            source.contains(&item.source_formula),
            "unbound expression {}",
            item.source_formula
        );
        assert!(!item.checks.is_empty());
    }
    let catalog = algorithmic_gas_benchmarks::convergence_validation::inventory()
        .expect("source expression catalog");
    let coverage = algorithmic_gas_benchmarks::convergence_estimates::expression_coverage(
        &catalog,
        std::slice::from_ref(&suite),
    );
    assert_eq!(
        coverage["unbound_evidence"],
        serde_json::json!([]),
        "source token or owning-label mismatch"
    );
    if std::env::var_os("CLONING_ESTIMATE_COVERAGE_DIAGNOSTIC").is_some() {
        let missing: Vec<_> = coverage["chapters"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|chapter| chapter["chapter"] == 3)
            .flat_map(|chapter| chapter["expressions"].as_array().unwrap())
            .filter(|expression| {
                expression["requires_expression_evidence"] == true
                    && expression["disposition"] != "numerically_checked_under_recorded_hypotheses"
            })
            .collect();
        std::fs::write(
            "/tmp/chapter03-current-missing-estimates.json",
            serde_json::to_vec_pretty(&missing).unwrap(),
        )
        .unwrap();
    }
    let certificate = suite
        .evidence
        .iter()
        .find(|e| e.source_formula.contains("3.X4"))
        .expect("directed rational expansion certificate");
    assert_eq!(
        certificate.inputs["marginal_variance_certificate"]["complete_patterns"],
        81
    );
    assert_eq!(
        certificate.inputs["marginal_variance_certificate"]["fixed_denominator"],
        "10^40"
    );
    assert!(
        suite
            .evidence
            .iter()
            .any(|e| e.inputs["alive"] == serde_json::json!([true, false, false, false]))
    );
    assert!(
        suite
            .evidence
            .iter()
            .any(|e| e.checks.iter().any(|c| c.id == "exact_efron_stein"))
    );
}

#[test]
fn invalid_sample_count_is_rejected_before_work() {
    assert!(futures_lite::future::block_on(validate_estimates(1)).is_err());
}
