"""Summarize executed, exactly source-bound estimate evidence and remaining gaps."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def summarize(report: dict, destination: Path) -> dict:
    """Write reproducible comparisons without granting credit for an inventory entry."""
    destination.mkdir(parents=True, exist_ok=True)
    coverage = report["coverage"]
    records = []
    failures = []
    for suite_index, suite in enumerate(report["suites"]):
        for evidence_index, evidence in enumerate(suite["evidence"]):
            base = {
                "report_path": f"/suites/{suite_index}/evidence/{evidence_index}",
                "chapter": evidence["chapter"],
                "evidence_source_labels": evidence["source_labels"],
                "source_formula": evidence["source_formula"],
                "input_reference": evidence["inputs"],
                "evidence_scope": evidence["scope"],
            }
            records.append({
                **base,
                "checks": evidence["checks"],
                "hypothesis_checks": evidence["hypothesis_checks"],
            })
            for category in ("hypothesis_checks", "checks"):
                checks = evidence[category]
                if isinstance(checks, dict):
                    checks = [
                        report["comparisons"][index]
                        for index in report["comparison_sets"][checks["set"]]
                    ]
                for comparison in checks:
                    entry = {**base, "comparison_kind": category, **comparison}
                    if not comparison["passed"]:
                        failures.append(entry)
    # Keep the report's exact relational representation. Expanding a large
    # finite-support fixture and source equation once per comparison can turn
    # a 200-MiB report into tens of GiB without adding information.
    ledger = {"schema_version": report.get("schema_version", 1), "records": records}
    if report.get("schema_version") == 2:
        ledger.update({
            name: report[name] for name in ("fixtures", "comparisons", "comparison_sets")
        })
    gaps = []
    groups = defaultdict(Counter)
    chapter_counts = {}
    for chapter in coverage["chapters"]:
        counts = Counter()
        for expression in chapter["expressions"]:
            if not expression.get("requires_expression_evidence", True):
                continue
            disposition = expression["disposition"]
            counts[disposition] += 1
            if disposition != "numerically_checked_under_recorded_hypotheses":
                gaps.append({"chapter": chapter["chapter"], **expression})
                groups[chapter["chapter"], expression.get("source_label")][disposition] += 1
        chapter_counts[chapter["chapter"]] = dict(counts)
    complete = (
        not gaps
        and not failures
        and not coverage["unbound_evidence"]
        and not report.get("phase_errors")
    )
    summary = {
        **report["summary"],
        "samples": report["samples"],
        "complete": complete,
        "chapter_required_expression_counts": chapter_counts,
        "unbound_evidence_records": len(coverage["unbound_evidence"]),
        "unvalidated_required_expressions": len(gaps),
        "source_hashes": {
            chapter["source"]: chapter["source_sha256"] for chapter in coverage["chapters"]
        },
    }
    for name, value in (
        ("summary.json", summary),
        ("comparisons.json", ledger),
        ("missing-estimates.json", gaps),
        ("failed-comparisons.json", failures),
        ("unbound-evidence.json", coverage["unbound_evidence"]),
    ):
        with (destination / name).open("w") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
    lines = [
        "# Individual convergence estimate validation",
        "",
        (
            f"Complete: **{complete}**. Independent sample count per configured experiment: "
            f"**{report['samples']}**."
        ),
        "",
        (
            f"Executed formula records: **{summary['formula_evidence_records']}**; numerical "
            f"comparisons: **{summary['comparisons']}**; failed comparisons: "
            f"**{summary['comparisons_failed']}**; failed hypotheses: "
            f"**{summary['hypotheses_failed']}**."
        ),
        "",
        (
            "An executed record validates only its exact quoted expression, for its recorded "
            "inputs and hypotheses. Every comparison retains the measured quantity, bound, "
            "slack and source. Finite experiments do not establish a universal analytic claim."
        ),
        "",
        "| Chapter | Executed expressions | Missing expressions | Analytic obligations |",
        "|---|---:|---:|---:|",
    ]
    for chapter, counts in chapter_counts.items():
        lines.append(
            f"| {chapter} | {counts.get('numerically_checked_under_recorded_hypotheses', 0)} "
            f"| {counts.get('no_expression_level_evidence', 0)} "
            f"| {counts.get('analytic_obligation', 0)} |"
        )
    lines += [
        "",
        (
            "The [comparison ledger](comparisons.json) contains the executed values. "
            "[Missing estimates](missing-estimates.json) retain the complete source equations "
            "and hypotheses; [failed comparisons](failed-comparisons.json) and "
            "[unbound evidence](unbound-evidence.json) cannot count toward completion."
        ),
        "",
        "| Chapter | Source label | Remaining required expressions |",
        "|---|---|---:|",
    ]
    for (chapter, label), counts in sorted(
        groups.items(), key=lambda item: (item[0][0], -sum(item[1].values()), item[0][1] or "")
    ):
        lines.append(f"| {chapter} | {label or '(source paragraph)'} | {sum(counts.values())} |")
    (destination / "summary.md").write_text("\n".join(lines) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(json.loads(args.report.read_text()), args.output), indent=2))
