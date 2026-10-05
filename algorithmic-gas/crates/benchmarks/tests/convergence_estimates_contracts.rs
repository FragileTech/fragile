use algorithmic_gas_benchmarks::{
    convergence_estimates::{EstimateSuite, expression_coverage},
    convergence_estimates_contracts::{append_cloning, append_framework, barrier, cutoff},
    convergence_validation::inventory,
};

#[test]
fn actual_integral_cutoff_and_barrier_collars_bind_and_pass() {
    let mut suite = EstimateSuite {
        chapter: 3,
        evidence: vec![],
        scope_notes: vec![],
    };
    append_cloning(&mut suite).unwrap();
    assert!(suite.evidence.len() > 40);
    for evidence in &suite.evidence {
        assert!(
            evidence.passed(),
            "{}: {:?}",
            evidence.source_formula,
            evidence.checks
        );
    }
    if let Ok(path) = std::env::var("FRAGILE_CONTRACT_EVIDENCE") {
        serde_json::to_writer(std::fs::File::create(path).unwrap(), &suite).unwrap();
    }
    let coverage = expression_coverage(&inventory().unwrap(), &[suite]);
    assert!(coverage["unbound_evidence"].as_array().unwrap().is_empty());
    let owned = [
        "lem-V-coercive",
        "lem-wasserstein-decomposition",
        "lem-fitness-gradient-boundary",
        "cor-extinction-suppression",
        "prop-barrier-existence",
        "lem-potential-bounds",
        "thm-derivation-of-stability-condition",
        "thm-stability-condition-final-corrected",
        "lem-unfit-fraction-lower-bound",
        "thm-unfit-high-error-overlap-fraction",
    ];
    for expression in coverage["chapters"]
        .as_array()
        .unwrap()
        .iter()
        .find(|chapter| chapter["chapter"] == 3)
        .unwrap()["expressions"]
        .as_array()
        .unwrap()
    {
        if expression["requires_expression_evidence"] == true
            && expression["source_label"]
                .as_str()
                .is_some_and(|label| owned.contains(&label))
        {
            assert_eq!(
                expression["disposition"], "numerically_checked_under_recorded_hypotheses",
                "Unvalidated owned source estimate: {}",
                expression["formula"]
            );
        }
    }
    assert_eq!(cutoff(1.), 1.);
    assert_eq!(cutoff(2.), 0.);
    assert!((cutoff(1.5) - 0.5).abs() < 1e-12);
    assert_eq!(barrier(0.0625, 0.125), 16.);
    assert_eq!(barrier(1., 0.125), 8.);
}

#[test]
fn quotient_state_contracts_agree_with_exact_transport_and_generator_moments() {
    let mut suite = EstimateSuite {
        chapter: 1,
        evidence: vec![],
        scope_notes: vec![],
    };
    append_framework(&mut suite).unwrap();
    assert!(suite.evidence.len() > 40);
    for evidence in &suite.evidence {
        assert!(
            evidence.passed(),
            "{}: {:?}",
            evidence.source_formula,
            evidence.checks
        );
    }
    let coverage = expression_coverage(&inventory().unwrap(), &[suite]);
    assert!(coverage["unbound_evidence"].as_array().unwrap().is_empty());
}
