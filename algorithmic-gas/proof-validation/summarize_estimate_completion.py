"""Present exact required-expression operands from a completed strict report."""

import argparse
import hashlib
import json
from pathlib import Path


def present(report_path: Path, output: Path) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report["summary"]["complete_required_estimates"]:
        message = "The required-expression completion gate did not pass"
        raise ValueError(message)
    output.mkdir(parents=True, exist_ok=False)
    digest = hashlib.sha256()
    with report_path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    catalog = {
        "report": str(report_path.resolve()),
        "report_sha256": digest.hexdigest(),
        "summary": report["summary"],
        "scope": (
            "Every required source expression has explicit evidence under recorded hypotheses. "
            "Operand ranges summarize its checks and are not new checks. Counts across source "
            "expressions overlap when an exact intermediate supports more than one expression."
        ),
        "chapters": [],
    }
    for chapter in report["coverage"]["chapters"]:
        rows = []
        for expression in chapter["expressions"]:
            if not expression["requires_expression_evidence"]:
                continue
            references = expression["evidence"]
            if expression["disposition"] != "numerically_checked_under_recorded_hypotheses":
                raise ValueError(f"Unchecked expression: {expression['id']}")
            operands = []
            scopes = set()
            for reference in references:
                evidence = report["suites"][reference["suite"]]["evidence"][reference["evidence"]]
                scopes.add(evidence["scope"])
                operands.extend(
                    report["comparisons"][index]
                    for index in report["comparison_sets"][evidence["checks"]["set"]]
                )
            row = {**expression, "recorded_scopes": sorted(scopes)}
            row["operand_ranges"] = (
                {
                    field: [min(c[field] for c in operands), max(c[field] for c in operands)]
                    for field in ("observed", "bound", "slack")
                }
                if operands
                else {}
            )
            row["comparison_ids"] = sorted({c["id"] for c in operands})
            rows.append(row)
        record = {
            "chapter": chapter["chapter"],
            "source": chapter["source"],
            "source_sha256": chapter["source_sha256"],
            "required_expressions_checked": len(rows),
            "expressions": rows,
        }
        name = f"chapter{chapter['chapter']:02d}-required-estimates"
        (output / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")
        lines = [
            f"# Chapter {chapter['chapter']}: every required quantitative expression",
            "",
            catalog["scope"],
            "",
            (
                "The JSON table retains complete source text, source lines, evidence indices, "
                "comparison IDs and all recorded scopes. Observed/bound ranges are across the "
                "expression's attached finite fixtures; matched pairs remain in the original report."
            ),
            "",
            "| Expression | Source label | Formula | Observed range | Bound range | Result |",
            "|---|---|---|---|---|---|",
        ]
        for row in rows:
            formula = " ".join(row["formula"].split()).replace("|", "&#124;")
            operands = row["operand_ranges"]
            fields = [str(operands.get(key, [])) for key in ("observed", "bound")]
            lines.append(
                f"| {row['id']} | {row['source_label']} | `{formula}` | "
                f"{fields[0]} | {fields[1]} | Pass under recorded hypotheses |"
            )
        (output / f"{name}.md").write_text("\n".join(lines) + "\n")
        catalog["chapters"].append({k: v for k, v in record.items() if k != "expressions"})
    (output / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
    return catalog


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = present(args.report, args.output)
    print(json.dumps({"chapters": result["chapters"], "summary": result["summary"]}))
