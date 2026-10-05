"""Render exact numeric record locations from the immutable Chapter 4 ledger."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LEDGER = (
    ROOT
    / "algorithmic-gas/outputs/convergence/chapters04-06-experiments/completion-20261004/statement-ledger.json"
)
OUTPUT = ROOT / "algorithmic-gas/proof-validation/chapter04-statement-results.md"
SOURCE = ROOT / "docs/source/2_fractal_gas/convergence_program/04_wasserstein_contraction.md"


def path_for(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def number(value: float) -> str:
    return f"{value:.9g}"


def main() -> None:
    data = json.loads(LEDGER.read_text())
    digest = hashlib.sha256(LEDGER.read_bytes()).hexdigest()
    lines = [
        "# Chapter 4 statement results and exact record locations",
        "",
        "This presentation adds zero numerical checks or simulation steps. It exposes the existing",
        "immutable statement ledger, including source locations, numerical operands and record IDs.",
        "A numerical subestimate does not establish the complete theorem, its universal hypotheses,",
        "existence/uniqueness or stationary-law conclusions. The seven unresolved entries remain visible.",
        "",
        f"Ledger SHA256: `{digest}`. Source SHA256: `{data['source_sha256']}`.",
        "",
        "Operands below use the retained comparison convention. A source lower bound may be stored",
        "with both operands negated to use an upper-bound comparison. The source expression, original",
        "hypotheses, units and scope are retained in each linked report. Record IDs locate the evidence",
        "inside the report; report-group totals include repeated finite cases, not independent seeds.",
        "",
        "| Source statement | Numerical evidence count | Example operands and exact record | Remaining scope |",
        "|---|---:|---|---|",
    ]
    for row in data["statement_ledger"]:
        groups = row["numerical_evidence_groups"]
        count = sum(group["numerical_comparisons"] for group in groups)
        label = row["source_label"]
        source_link = f"[{label}]({SOURCE}:{row['source_line']})"
        title = row["title"].replace("|", r"\|")
        if groups:
            examples = []
            ordered = sorted(
                groups, key=lambda group: "completion-20261004" not in group["report_path"]
            )
            for group in ordered:
                record = group["worst_record"]
                op = "=" if record.get("relation") == "equal" else "≤"
                report = path_for(group["report_path"])
                examples.append(
                    f"{number(record['observed'])} {op} {number(record['bound'])}: [`{record['id']}`]({report})"
                )
            remaining = "Finite subestimates; universal/imported hypotheses remain analytic."
            example = "; ".join(examples)
        else:
            example = "No applicable numerical theorem conclusion credited."
            remaining = row["remaining_obligation"].replace("|", r"\|").replace("\n", " ")
        lines.append(f"| {source_link}: {title} | {count} | {example} | {remaining} |")
    lines += [
        "",
        "The [complete estimate table](chapter04-estimate-completion.md) gives the dimensions, normalization,",
        "full pressure integration, native reference determinant/survival checks, logarithmic QSD primitives",
        "and their numerical usefulness. The [full statement ledger]("
        + str(LEDGER)
        + ") retains every evidence",
        "group and exact complete source statement, rather than only the examples displayed here.",
        "",
        "The [corrected original-run presentation]("
        + str(
            ROOT / "algorithmic-gas/outputs/convergence/chapter04-full-corrected-presentation.json"
        )
        + ")",
        "and [reference force audit]("
        + str(ROOT / "algorithmic-gas/outputs/convergence/chapter04-full-reference-audit.json")
        + ")",
        "retain the exact per-timestep predicates and continuous-trajectory applicability corrections.",
        "Their derived status does not overwrite the original experiment report or raw native archives.",
        "",
    ]
    OUTPUT.write_text("\n".join(lines))
    print(
        json.dumps({
            "presentation": str(OUTPUT),
            "statements": len(data["statement_ledger"]),
            "new_numeric_checks": 0,
            "ledger_sha256": digest,
        })
    )


if __name__ == "__main__":
    main()
