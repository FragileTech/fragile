use algorithmic_gas_benchmarks::convergence_estimates_kinetic::validate_estimates;

#[test]
fn chapter_two_estimates_bind_exact_source_and_keep_numerical_evidence() {
    futures_lite::future::block_on(async {
        let suite = validate_estimates(128)
            .await
            .expect("chapter 2 estimate suite");
        assert_eq!(suite.chapter, 2);
        assert!(
            suite.evidence.len() > 100,
            "every source formula has its own evidence"
        );
        let failed = suite
            .evidence
            .iter()
            .filter(|e| !e.passed())
            .map(|e| {
                (
                    &e.source_labels,
                    &e.source_formula,
                    e.checks.iter().filter(|c| !c.passed).collect::<Vec<_>>(),
                    e.hypothesis_checks
                        .iter()
                        .filter(|c| !c.passed)
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>();
        assert!(failed.is_empty(), "failed evidence: {failed:#?}");
        let source = include_str!(
            "../../../../docs/source/2_fractal_gas/convergence_program/02_euclidean_gas.md"
        );
        for e in &suite.evidence {
            assert!(source.contains(&e.source_formula));
            assert!(!e.checks.is_empty());
            assert!(!e.inputs.is_null());
            assert!(!e.scope.is_empty());
        }
        let catalog: serde_json::Value = serde_json::from_str(include_str!(
            "../../../proof-validation/chapter02_inventory.json"
        ))
        .unwrap();
        let coverage = algorithmic_gas_benchmarks::convergence_estimates::expression_coverage(
            &[catalog],
            std::slice::from_ref(&suite),
        );
        let expressions = coverage["chapters"][0]["expressions"].as_array().unwrap();
        let missing_bounds = expressions
            .iter()
            .filter(|expression| {
                expression["expression_role"] == "quantitative_bound"
                    && expression["disposition"] != "numerically_checked_under_recorded_hypotheses"
            })
            .map(|expression| &expression["formula"])
            .collect::<Vec<_>>();
        assert!(
            missing_bounds.is_empty(),
            "unvalidated chapter 2 quantitative source bounds: {missing_bounds:?}"
        );
        assert!(coverage["unbound_evidence"].as_array().unwrap().is_empty());
        if let Ok(path) = std::env::var("GAS_CHAPTER02_COVERAGE_OUTPUT") {
            std::fs::write(path, serde_json::to_vec_pretty(&coverage).unwrap()).unwrap();
        }
        if let Ok(path) = std::env::var("GAS_CHAPTER02_EVIDENCE_OUTPUT") {
            std::fs::write(path, serde_json::to_vec_pretty(&suite).unwrap()).unwrap();
        }
    });
}
#[test]
fn replica_configuration_rejects_unusable_uncertainty_estimates() {
    assert!(futures_lite::future::block_on(validate_estimates(1)).is_err());
}

#[test]
fn chapter_two_display_delimiters_never_capture_prose() {
    let source = include_str!(
        "../../../../docs/source/2_fractal_gas/convergence_program/02_euclidean_gas.md"
    );
    let lines = source.lines().collect::<Vec<_>>();
    let delimiters = lines
        .iter()
        .enumerate()
        .filter_map(|(i, line)| (line.trim() == "$$").then_some(i))
        .collect::<Vec<_>>();
    assert_eq!(
        delimiters.len() % 2,
        0,
        "unmatched standalone math delimiter"
    );
    for pair in delimiters.chunks_exact(2) {
        let body = lines[pair[0] + 1..pair[1]].join("\n");
        for prose_marker in [":label:", "*Proof.*", "```{dropdown}", ":::{prf:"] {
            assert!(
                !body.contains(prose_marker),
                "prose captured as math at line {}",
                pair[0] + 1
            );
        }
    }
}
