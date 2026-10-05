//! Source-bound coverage for analyses of previously recorded native experiments.
use algorithmic_gas::{GasError, Result};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

fn compact(text: &str) -> String {
    text.chars().filter(|c| !c.is_whitespace()).collect()
}

fn strings(value: &Value) -> Vec<&str> {
    value
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect()
}

/// Bind only whole source expressions to passed finite numerical comparisons.
/// A theorem label, a source fragment or an applicability assertion alone earns
/// no expression credit. Missing expressions remain in the saved coverage ledger.
pub fn attach_coverage(report: &mut Value, inventory: &Value, source: &str) -> Result<()> {
    let chapter = inventory["chapter"]
        .as_u64()
        .ok_or_else(|| GasError::Configuration("inventory chapter missing".into()))?;
    if let Some(worker_coverage) = report.get("coverage").cloned() {
        report["worker_coverage"] = worker_coverage;
    }
    if report["chapter"].as_u64() != Some(chapter) {
        return Err(GasError::Configuration(
            "report/inventory chapter mismatch".into(),
        ));
    }
    let expressions = inventory["quantitative_expressions"]
        .as_array()
        .ok_or_else(|| GasError::Configuration("inventory expressions missing".into()))?;
    let comparisons = report["comparisons"]
        .as_array()
        .ok_or_else(|| GasError::Configuration("stored comparisons missing".into()))?;
    let source_compact = compact(source);
    let mut expression_index: BTreeMap<String, Vec<&Value>> = BTreeMap::new();
    for expression in expressions {
        expression_index
            .entry(compact(expression["formula"].as_str().unwrap_or("")))
            .or_default()
            .push(expression);
    }
    let mut evidence: BTreeMap<String, Vec<Value>> = BTreeMap::new();
    let mut unbound = vec![];
    for (index, comparison) in comparisons.iter().enumerate() {
        let mut quotes = strings(&comparison["source_quotes"]);
        if let Some(formula) = comparison["source_formula"].as_str() {
            quotes.push(formula);
        }
        let labels: BTreeSet<_> = strings(&comparison["source_labels"]).into_iter().collect();
        let eligible = comparison["passed"] == true
            && comparison["observed"].as_f64().is_some_and(f64::is_finite)
            && comparison["bound"].as_f64().is_some_and(f64::is_finite);
        for quote in quotes {
            let key = compact(quote);
            if key.is_empty() || !source_compact.contains(&key) {
                unbound.push(
                    json!({"comparison_index":index,"id":comparison["id"],"source_quote":quote}),
                );
                continue;
            }
            if !eligible {
                continue;
            }
            for expression in expression_index.get(&key).into_iter().flatten() {
                if expression["source_label"]
                    .as_str()
                    .is_none_or(|label| labels.contains(label))
                {
                    evidence.entry(expression["id"].as_str().unwrap_or("").into())
                        .or_default().push(json!({"comparison_index":index,"id":comparison["id"],"scope":comparison["scope"]}));
                }
            }
        }
    }
    let mut ledger = vec![];
    let mut required = 0;
    let mut covered_required = 0;
    for expression in expressions {
        let id = expression["id"].as_str().unwrap_or("");
        let required_here = expression["requires_expression_evidence"] == true;
        let covered = evidence.contains_key(id);
        required += usize::from(required_here);
        covered_required += usize::from(required_here && covered);
        ledger.push(json!({
            "expression_id":id,"source_line":expression["source_line"],
            "source_label":expression["source_label"],"formula":expression["formula"],
            "requires_expression_evidence":required_here,
            "status":if covered {"checked_for_recorded_inputs"} else if required_here {"not_checked_from_stored_data"} else {"domain_or_symbol_reference"},
            "evidence":evidence.get(id),
        }));
    }
    report["coverage"] = json!({
        "source":inventory["source"],"source_sha256":inventory["source_sha256"],
        "required_expressions":required,"checked_required_expressions":covered_required,
        "missing_required_expressions":required-covered_required,
        "complete":required==covered_required && unbound.is_empty(),
        "unbound_evidence":unbound,"expression_ledger":ledger,
        "scope":"Whole-expression finite numerical evidence only. Global hypotheses and law convergence require their separate certificates."
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::attach_coverage;
    use serde_json::json;

    fn inventory() -> serde_json::Value {
        json!({"chapter":4,"source":"fixture","source_sha256":"fixture","quantitative_expressions":[
            {"id":"a","formula":"x \\le y","source_label":"lemma","requires_expression_evidence":true}
        ]})
    }

    #[test]
    fn labels_and_fragments_do_not_grant_expression_credit() {
        let mut report = json!({"chapter":4,"comparisons":[{"id":"fragment","source_labels":["lemma"],"source_formula":"x","observed":1.,"bound":2.,"passed":true}]});
        attach_coverage(&mut report, &inventory(), "x \\le y").unwrap();
        assert_eq!(report["coverage"]["checked_required_expressions"], 0);
    }

    #[test]
    fn failed_or_non_numeric_evidence_cannot_complete_coverage() {
        let mut report = json!({"chapter":4,"comparisons":[{"id":"failed","source_labels":["lemma"],"source_formula":"x \\le y","observed":3.,"bound":2.,"passed":false}]});
        attach_coverage(&mut report, &inventory(), "x \\le y").unwrap();
        assert_eq!(report["coverage"]["complete"], false);
        report["comparisons"][0]["passed"] = json!(true);
        report["comparisons"][0]["observed"] = serde_json::Value::Null;
        attach_coverage(&mut report, &inventory(), "x \\le y").unwrap();
        assert_eq!(report["coverage"]["complete"], false);
    }

    #[test]
    fn complete_owned_expression_is_counted_once() {
        let check = json!({"id":"check","source_labels":["lemma"],"source_formula":"x \\le y","observed":1.,"bound":2.,"passed":true});
        let mut report = json!({"chapter":4,"comparisons":[check.clone(),check]});
        attach_coverage(&mut report, &inventory(), "x \\le y").unwrap();
        assert_eq!(report["coverage"]["checked_required_expressions"], 1);
        assert_eq!(report["coverage"]["complete"], true);
    }
}
