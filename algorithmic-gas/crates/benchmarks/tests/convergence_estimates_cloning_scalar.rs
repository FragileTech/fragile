use algorithmic_gas_benchmarks::{
    convergence_estimates::EstimateSuite,
    convergence_estimates_cloning_scalar::append_cloning_scalar,
};

#[test]
fn scalar_proof_identities_retain_exact_quotes_and_actual_hypotheses() {
    let mut suite = EstimateSuite {
        chapter: 3,
        evidence: vec![],
        scope_notes: vec![],
    };
    append_cloning_scalar(&mut suite).expect("cloning scalar supplement");
    if let Ok(path) = std::env::var("GAS_CLONING_SCALAR_EVIDENCE_OUTPUT") {
        std::fs::write(path, serde_json::to_vec(&suite).unwrap()).unwrap();
    }
    assert!(suite.evidence.len() >= 150);
    let failed = suite
        .evidence
        .iter()
        .filter(|record| !record.passed())
        .map(|record| {
            (
                &record.source_labels,
                &record.source_formula,
                record
                    .checks
                    .iter()
                    .chain(&record.hypothesis_checks)
                    .filter(|check| !check.passed)
                    .take(12)
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();
    assert!(failed.is_empty(), "scalar numerical failures: {failed:#?}");
    let source =
        include_str!("../../../../docs/source/2_fractal_gas/convergence_program/03_cloning.md");
    for record in &suite.evidence {
        assert!(source.contains(&record.source_formula));
        assert!(!record.checks.is_empty());
        assert!(!record.inputs["cases"].as_array().unwrap().is_empty());
    }
}

#[test]
fn scalar_supplement_rejects_another_chapter() {
    let mut suite = EstimateSuite {
        chapter: 2,
        evidence: vec![],
        scope_notes: vec![],
    };
    assert!(append_cloning_scalar(&mut suite).is_err());
}
