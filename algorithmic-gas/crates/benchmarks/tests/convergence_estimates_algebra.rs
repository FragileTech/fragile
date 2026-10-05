use algorithmic_gas_benchmarks::{
    convergence_estimates::{EstimateSuite, expression_coverage},
    convergence_estimates_algebra::append_framework,
    convergence_validation::inventory,
};

#[test]
fn elementary_chains_and_full_density_comparisons_pass_and_bind_exact_sources() {
    let mut suite = EstimateSuite {
        chapter: 1,
        evidence: vec![],
        scope_notes: vec![],
    };
    append_framework(&mut suite).unwrap();
    assert!(suite.evidence.len() > 100);
    for evidence in &suite.evidence {
        assert!(
            evidence.passed(),
            "{} {:?}",
            evidence.source_formula,
            evidence.checks
        );
    }
    let coverage = expression_coverage(&inventory().unwrap(), &[suite]);
    assert!(coverage["unbound_evidence"].as_array().unwrap().is_empty());
}
