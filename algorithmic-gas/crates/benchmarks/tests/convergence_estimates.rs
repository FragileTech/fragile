use algorithmic_gas_benchmarks::{
    convergence_estimates::{
        EstimateEvidence, EstimateSuite, compact_evidence, expression_coverage,
        required_estimates_complete, validate_chapters,
    },
    convergence_framework::BoundCheck,
};
use serde_json::json;

#[test]
fn invalid_chapter_selection_is_rejected_before_experiments() {
    for chapters in [vec![], vec![0], vec![4], vec![1, 1]] {
        assert!(futures_lite::future::block_on(validate_chapters(32, &chapters)).is_err());
    }
}

fn suite(formula: &str, labels: &[&str], checks: Vec<BoundCheck>) -> EstimateSuite {
    EstimateSuite {
        chapter: 1,
        evidence: vec![EstimateEvidence {
            chapter: 1,
            source_labels: labels.iter().map(|label| (*label).to_owned()).collect(),
            source_formula: formula.to_owned(),
            inputs: json!({"x":0.25}),
            hypothesis_checks: vec![BoundCheck::upper("domain", labels, "x in [0,1]", 0.25, 1.)],
            checks,
            scope: "Declared finite fixture; no universal theorem inferred.".to_owned(),
        }],
        scope_notes: vec![],
    }
}

#[test]
fn shared_fixtures_and_comparisons_retain_exact_evidence_references() {
    let mut fixture = suite(
        "x^2 \\le x",
        &["lemma"],
        vec![BoundCheck::upper("square", &["lemma"], "domain", 0.25, 0.5)],
    );
    let mut other = fixture.evidence[0].clone();
    other.source_formula = "x \\le 1".to_owned();
    fixture.evidence.push(other);
    let report = compact_evidence(&[fixture]).unwrap();
    assert_eq!(report["fixtures"].as_array().unwrap().len(), 1);
    assert_eq!(report["comparisons"].as_array().unwrap().len(), 2);
    let left = &report["suites"][0]["evidence"][0];
    let right = &report["suites"][0]["evidence"][1];
    assert_eq!(left["inputs"], right["inputs"]);
    assert_eq!(left["checks"], right["checks"]);
    let set = left["checks"]["set"].as_u64().unwrap() as usize;
    let comparison = report["comparison_sets"][set][0].as_u64().unwrap() as usize;
    assert_eq!(report["comparisons"][comparison]["observed"], 0.25);
    assert_eq!(report["comparisons"][comparison]["bound"], 0.5);
}

#[test]
fn analytic_obligations_and_unbound_records_do_not_satisfy_completion() {
    let mut coverage = json!({
        "required_expression_counts":{"numerically_checked_under_recorded_hypotheses":3},
        "unbound_evidence":[]
    });
    assert!(required_estimates_complete(&coverage));
    coverage["required_expression_counts"]["analytic_obligation"] = json!(1);
    assert!(!required_estimates_complete(&coverage));
    coverage["required_expression_counts"]["analytic_obligation"] = json!(0);
    coverage["unbound_evidence"] = json!([{"source_formula":"unquoted"}]);
    assert!(!required_estimates_complete(&coverage));
    assert!(!required_estimates_complete(&json!({})));
}

fn catalog() -> Vec<serde_json::Value> {
    vec![json!({
        "chapter":1,"source":"fixture","source_sha256":"fixture",
        "quantitative_expressions":[
            {"id":"bound","source_label":"lemma","formula":"x^2 \\le x",
             "computability_kind":"exact","expression_role":"quantitative_bound",
             "requires_expression_evidence":true},
            {"id":"same_label_untested","source_label":"lemma","formula":"y^2 \\le y",
             "computability_kind":"exact","requires_expression_evidence":true},
            {"id":"other_scope","source_label":"other","formula":"x^2 \\le x",
             "computability_kind":"exact","requires_expression_evidence":true}
        ]
    })]
}

#[test]
fn evidence_binds_exact_tokens_and_owner_without_promoting_a_whole_label() {
    let evidence = suite(
        " x^2\n\\le  x ",
        &["lemma"],
        vec![BoundCheck::upper(
            "square",
            &["lemma"],
            "x in [0,1]",
            0.25_f64.powi(2),
            0.25,
        )],
    );
    let coverage = expression_coverage(&catalog(), &[evidence]);
    let expressions = coverage["chapters"][0]["expressions"].as_array().unwrap();
    assert_eq!(
        expressions[0]["disposition"],
        "numerically_checked_under_recorded_hypotheses"
    );
    assert_eq!(
        expressions[1]["disposition"],
        "no_expression_level_evidence"
    );
    assert_eq!(
        expressions[2]["disposition"],
        "no_expression_level_evidence"
    );
    assert_eq!(
        coverage["required_expression_counts"]["numerically_checked_under_recorded_hypotheses"],
        1
    );
    assert!(coverage["unbound_evidence"].as_array().unwrap().is_empty());
}

#[test]
fn partial_quotes_empty_comparisons_and_failed_hypotheses_do_not_validate() {
    let partial = suite(
        "x^2",
        &["lemma"],
        vec![BoundCheck::upper("square", &["lemma"], "domain", 0., 1.)],
    );
    let coverage = expression_coverage(&catalog(), &[partial]);
    assert_eq!(coverage["unbound_evidence"].as_array().unwrap().len(), 1);
    let mut empty = suite("x^2 \\le x", &["lemma"], vec![]);
    assert!(!empty.evidence[0].passed());
    empty.evidence[0]
        .checks
        .push(BoundCheck::upper("square", &["lemma"], "domain", 0., 1.));
    empty.evidence[0].hypothesis_checks[0].passed = false;
    let coverage = expression_coverage(&catalog(), &[empty]);
    assert_eq!(
        coverage["chapters"][0]["expressions"][0]["disposition"],
        "comparison_failed_or_hypotheses_unmet"
    );
}

#[test]
fn one_passing_fixture_cannot_hide_a_failed_fixture_of_the_same_estimate() {
    let good = suite(
        "x^2 \\le x",
        &["lemma"],
        vec![BoundCheck::upper("square", &["lemma"], "domain", 0.25, 0.5)],
    );
    let bad = suite(
        "x^2 \\le x",
        &["lemma"],
        vec![BoundCheck::upper("square", &["lemma"], "domain", 1., 0.5)],
    );
    let coverage = expression_coverage(&catalog(), &[good, bad]);
    assert_eq!(
        coverage["chapters"][0]["expressions"][0]["disposition"],
        "comparison_failed_or_hypotheses_unmet"
    );
    assert!(!required_estimates_complete(&coverage));
}
