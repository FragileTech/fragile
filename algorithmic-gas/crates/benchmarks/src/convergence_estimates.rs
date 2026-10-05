//! Formula-level evidence for the first three convergence chapters.
//!
//! A source label alone never validates the estimates inside its statement or
//! proof. Each evidence record quotes the expression actually checked, retains
//! its inputs and hypotheses, and stores the numerical comparisons separately.
use crate::convergence_framework::BoundCheck;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, hash_map::DefaultHasher},
    hash::{Hash, Hasher},
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EstimateEvidence {
    pub chapter: u8,
    pub source_labels: Vec<String>,
    /// Exact source expression. Whitespace normalization is allowed when the
    /// catalog binds this record; changing its mathematical tokens is not.
    pub source_formula: String,
    pub inputs: Value,
    pub hypothesis_checks: Vec<BoundCheck>,
    pub checks: Vec<BoundCheck>,
    pub scope: String,
}

impl EstimateEvidence {
    pub fn passed(&self) -> bool {
        !self.checks.is_empty()
            && self.checks.iter().all(|check| check.passed)
            && self.hypothesis_checks.iter().all(|check| check.passed)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EstimateSuite {
    pub chapter: u8,
    pub evidence: Vec<EstimateEvidence>,
    pub scope_notes: Vec<String>,
}

pub async fn validate_all(samples: usize) -> algorithmic_gas::Result<Value> {
    validate_chapters(samples, &[1, 2, 3]).await
}

/// Run selected chapters, retaining completed phases if another phase fails.
pub async fn validate_chapters(samples: usize, selected: &[u8]) -> algorithmic_gas::Result<Value> {
    if !(32..=100_000).contains(&samples) {
        return Err(algorithmic_gas::GasError::Configuration(
            "estimate experiments require 32..100000 independent samples".to_owned(),
        ));
    }
    let chapters = selected.iter().copied().collect::<BTreeSet<_>>();
    if chapters.is_empty()
        || chapters.len() != selected.len()
        || chapters.iter().any(|c| !(1..=3).contains(c))
    {
        return Err(algorithmic_gas::GasError::Configuration(
            "select distinct convergence chapters from 1,2,3".to_owned(),
        ));
    }
    let mut suites = Vec::new();
    let mut phase_errors = Vec::new();
    for chapter in &chapters {
        let result = match chapter {
            1 => crate::convergence_estimates_framework::validate_estimates(samples).await,
            2 => crate::convergence_estimates_kinetic::validate_estimates(samples).await,
            _ => crate::convergence_estimates_cloning::validate_estimates(samples).await,
        };
        match result {
            Ok(mut suite) => {
                if *chapter == 1
                    && let Err(error) =
                        crate::convergence_estimates_algebra::append_framework(&mut suite)
                {
                    phase_errors.push(serde_json::json!({"chapter":1,"phase":"algebra","error":error.to_string()}));
                }
                if *chapter == 1
                    && let Err(error) =
                        crate::convergence_estimates_contracts::append_framework(&mut suite)
                {
                    phase_errors.push(serde_json::json!({"chapter":1,"phase":"contracts","error":error.to_string()}));
                }
                if *chapter == 1
                    && let Err(error) =
                        crate::convergence_estimates_scalar::append_framework_scalar(&mut suite)
                {
                    phase_errors.push(
                        serde_json::json!({"chapter":1,"phase":"scalar","error":error.to_string()}),
                    );
                }
                if *chapter == 1
                    && let Err(error) =
                        crate::convergence_estimates_framework::append_remaining_framework(
                            &mut suite,
                        )
                {
                    phase_errors.push(serde_json::json!({"chapter":1,"phase":"remaining_framework","error":error.to_string()}));
                }
                if *chapter == 3
                    && let Err(error) =
                        crate::convergence_estimates_contracts::append_cloning(&mut suite)
                {
                    phase_errors.push(serde_json::json!({"chapter":3,"phase":"contracts","error":error.to_string()}));
                }
                if *chapter == 3
                    && let Err(error) =
                        crate::convergence_estimates_cloning_scalar::append_cloning_scalar(
                            &mut suite,
                        )
                {
                    phase_errors.push(
                        serde_json::json!({"chapter":3,"phase":"scalar","error":error.to_string()}),
                    );
                }
                suites.push(suite);
            }
            Err(error) => {
                phase_errors.push(serde_json::json!({"chapter":chapter,"error":error.to_string()}))
            }
        }
    }
    let catalog = crate::convergence_validation::inventory()?
        .into_iter()
        .filter(|chapter| {
            chapter["chapter"]
                .as_u64()
                .is_some_and(|number| chapters.contains(&(number as u8)))
        })
        .collect::<Vec<_>>();
    let checks = suites
        .iter()
        .flat_map(|suite| &suite.evidence)
        .flat_map(|evidence| &evidence.checks)
        .collect::<Vec<_>>();
    let hypotheses = suites
        .iter()
        .flat_map(|suite| &suite.evidence)
        .flat_map(|evidence| &evidence.hypothesis_checks)
        .collect::<Vec<_>>();
    let coverage = expression_coverage(&catalog, &suites);
    let complete = phase_errors.is_empty() && required_estimates_complete(&coverage);
    let encoded = compact_evidence(&suites)?;
    Ok(serde_json::json!({
        "schema_version":2,"samples":samples,"suites":encoded["suites"],
        "fixtures":encoded["fixtures"],"comparisons":encoded["comparisons"],
        "comparison_sets":encoded["comparison_sets"],"coverage":coverage,
        "selected_chapters":chapters,"phase_errors":phase_errors,
        "summary":{
            "formula_evidence_records":suites.iter().map(|suite|suite.evidence.len()).sum::<usize>(),
            "complete_required_estimates":complete,
            "phase_errors":phase_errors.len(),
            "comparisons":checks.len(),"comparisons_failed":checks.iter().filter(|check|!check.passed).count(),
            "hypothesis_comparisons":hypotheses.len(),"hypotheses_failed":hypotheses.iter().filter(|check|!check.passed).count()
        }
    }))
}

/// Store each input fixture and numerical comparison once. Formula records
/// retain exact references to their comparisons rather than repeating large
/// exhaustive finite-support tables for every step of the same proof.
pub fn compact_evidence(suites: &[EstimateSuite]) -> algorithmic_gas::Result<Value> {
    fn intern(
        value: Value,
        values: &mut Vec<Value>,
        indexes: &mut HashMap<u64, Vec<usize>>,
    ) -> usize {
        let mut hasher = DefaultHasher::new();
        value.hash(&mut hasher);
        let bucket = indexes.entry(hasher.finish()).or_default();
        // Hashes only choose a bucket. Exact equality remains mandatory, so
        // even a hash collision cannot silently merge distinct measurements.
        if let Some(index) = bucket.iter().find(|&&index| values[index] == value) {
            return *index;
        }
        let index = values.len();
        values.push(value);
        bucket.push(index);
        index
    }
    let mut fixtures = Vec::new();
    let mut fixture_indexes = HashMap::new();
    let mut comparisons = Vec::new();
    let mut comparison_indexes = HashMap::new();
    let mut comparison_sets = Vec::new();
    let mut set_indexes = HashMap::new();
    let mut encoded_suites = Vec::new();
    for suite in suites {
        let mut records = Vec::new();
        for record in &suite.evidence {
            let fixture = intern(record.inputs.clone(), &mut fixtures, &mut fixture_indexes);
            let mut sets = Vec::new();
            for checks in [&record.hypothesis_checks, &record.checks] {
                let mut indexes = Vec::new();
                for check in checks {
                    indexes.push(intern(
                        serde_json::json!(check),
                        &mut comparisons,
                        &mut comparison_indexes,
                    ));
                }
                sets.push(intern(
                    serde_json::json!(indexes),
                    &mut comparison_sets,
                    &mut set_indexes,
                ));
            }
            records.push(serde_json::json!({
                "chapter":record.chapter,"source_labels":record.source_labels,
                "source_formula":record.source_formula,"inputs":{"fixture":fixture},
                "hypothesis_checks":{"set":sets[0]},"checks":{"set":sets[1]},
                "scope":record.scope,"passed":record.passed()
            }));
        }
        encoded_suites.push(serde_json::json!({
            "chapter":suite.chapter,"evidence":records,"scope_notes":suite.scope_notes
        }));
    }
    Ok(
        serde_json::json!({"suites":encoded_suites,"fixtures":fixtures,
        "comparisons":comparisons,"comparison_sets":comparison_sets}),
    )
}

/// Completion requires individual evidence for every required expression.
/// An inventory entry or an unexecuted analytic obligation does not qualify.
pub fn required_estimates_complete(coverage: &Value) -> bool {
    coverage["unbound_evidence"]
        .as_array()
        .is_some_and(Vec::is_empty)
        && coverage["required_expression_counts"]
            .as_object()
            .is_some_and(|counts| {
                !counts.is_empty()
                    && counts.iter().all(|(status, count)| {
                        status == "numerically_checked_under_recorded_hypotheses"
                            || count.as_u64() == Some(0)
                    })
            })
}

fn mathematical_tokens(value: &str) -> String {
    value
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect()
}

/// Bind actual comparisons to exact source expressions. Matching a theorem
/// label, a substring, or a similarly named constant supplies no evidence.
pub fn expression_coverage(catalog: &[Value], suites: &[EstimateSuite]) -> Value {
    let mut chapters = Vec::new();
    let mut counts = BTreeMap::<String, usize>::new();
    let mut required_counts = BTreeMap::<String, usize>::new();
    let mut role_counts = BTreeMap::<String, usize>::new();
    let mut unbound_evidence = Vec::new();
    let mut bindings = BTreeMap::<(u64, String), Vec<Value>>::new();
    // Normalize each source equation once. Exhaustive categorical witnesses
    // produce many records for the same equation; rescanning and allocating
    // the entire source catalog for each record obscures the experiment cost.
    let mut source_index = BTreeMap::<(u64, String), Vec<&Value>>::new();
    for chapter in catalog {
        if let (Some(number), Some(expressions)) = (
            chapter["chapter"].as_u64(),
            chapter["quantitative_expressions"].as_array(),
        ) {
            for expression in expressions {
                if let Some(formula) = expression["formula"].as_str() {
                    source_index
                        .entry((number, mathematical_tokens(formula)))
                        .or_default()
                        .push(expression);
                }
            }
        }
    }
    for (suite_index, suite) in suites.iter().enumerate() {
        for (evidence_index, evidence) in suite.evidence.iter().enumerate() {
            let formula = mathematical_tokens(&evidence.source_formula);
            let candidates = source_index
                .get(&(evidence.chapter as u64, formula))
                .into_iter()
                .flatten()
                .copied()
                .filter(|expression| {
                    expression["source_label"].as_str().is_none_or(|label| {
                        evidence.source_labels.iter().any(|source| source == label)
                    })
                })
                .collect::<Vec<_>>();
            if candidates.is_empty() {
                unbound_evidence.push(serde_json::json!({
                    "suite":suite_index,"evidence":evidence_index,
                    "chapter":evidence.chapter,"source_labels":evidence.source_labels,
                    "source_formula":evidence.source_formula,
                    "reason":"No exact source expression matches; stale or partial quotes cannot validate an estimate."
                }));
            }
            for expression in candidates {
                if let Some(id) = expression["id"].as_str() {
                    bindings
                        .entry((evidence.chapter as u64, id.to_owned()))
                        .or_default()
                        .push(serde_json::json!({
                            "suite":suite_index,"evidence":evidence_index,
                            "comparison_count":evidence.checks.len(),
                            "hypothesis_comparison_count":evidence.hypothesis_checks.len(),
                            "passed":evidence.passed(),"scope":evidence.scope,
                            "source_labels":evidence.source_labels
                        }));
                }
            }
        }
    }
    for chapter in catalog {
        let number = chapter["chapter"].as_u64().unwrap_or(0);
        let mut expressions = Vec::new();
        for expression in chapter["quantitative_expressions"]
            .as_array()
            .into_iter()
            .flatten()
        {
            let id = expression["id"].as_str().unwrap_or("");
            let witnesses = bindings
                .get(&(number, id.to_owned()))
                .cloned()
                .unwrap_or_default();
            let disposition = if witnesses.iter().any(|witness| witness["passed"] != true) {
                "comparison_failed_or_hypotheses_unmet"
            } else if !witnesses.is_empty() {
                "numerically_checked_under_recorded_hypotheses"
            } else if expression["computability_kind"] == "not_numerically_testable" {
                "analytic_obligation"
            } else {
                "no_expression_level_evidence"
            };
            *counts.entry(disposition.to_owned()).or_default() += 1;
            let role = expression["expression_role"]
                .as_str()
                .unwrap_or("unclassified");
            *role_counts.entry(role.to_owned()).or_default() += 1;
            if expression["requires_expression_evidence"]
                .as_bool()
                .unwrap_or(true)
            {
                *required_counts.entry(disposition.to_owned()).or_default() += 1;
            }
            let mut indexed = expression.clone();
            if let Some(object) = indexed.as_object_mut() {
                object.insert("disposition".to_owned(), serde_json::json!(disposition));
                object.insert("evidence".to_owned(), serde_json::json!(witnesses));
            }
            expressions.push(indexed);
        }
        chapters.push(serde_json::json!({
            "chapter":number,"source":chapter["source"],"source_sha256":chapter["source_sha256"],
            "expressions":expressions
        }));
    }
    let evaluated_labels = suites
        .iter()
        .flat_map(|suite| &suite.evidence)
        .filter(|evidence| evidence.passed())
        .flat_map(|evidence| evidence.source_labels.iter().cloned())
        .collect::<BTreeSet<_>>();
    serde_json::json!({"expression_counts":counts,"required_expression_counts":required_counts,
        "expression_role_counts":role_counts,"chapters":chapters,
        "unbound_evidence":unbound_evidence,"evaluated_labels":evaluated_labels,
        "scope":"Each numerical status names an exact quoted expression, input witness and concrete comparisons. Finite checks retain their hypotheses; no label-level status validates unquoted expressions or a universal theorem."})
}
