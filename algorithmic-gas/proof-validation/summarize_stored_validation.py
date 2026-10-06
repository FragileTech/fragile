"""Summarize source-bound chapters 4–6 comparisons from retained experiments."""

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def cell(value):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.9g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def resolve_input(root: Path, original: str) -> Path:
    """Resolve current paths and retained reports written before extraction."""
    path = Path(original)
    if path.is_absolute():
        return path
    candidate = root / path
    if candidate.exists() or not path.parts or path.parts[0] != "algorithmic-gas":
        return candidate
    return root.joinpath(*path.parts[1:])


def plot_decay(report, destination, workspace_root=None):
    """Plot measured errors against the independently checked stored envelopes."""
    import matplotlib.pyplot as plt

    root = workspace_root or Path(__file__).resolve().parents[1]
    inputs = []
    for original in report["input_reports"]:
        path = Path(original)
        if path.name not in {"decay-shared.json", "decay-independent.json"}:
            continue
        path = resolve_input(root, original)
        inputs.append(json.loads(path.read_text()))
    for randomness, profile, name in (
        ("shared", "linear_quadratic_noiseless", "quadratic-contraction.png"),
        ("independent", "linear_quadratic_independent_noise", "quadratic-noise-floor.png"),
    ):
        cases = [
            case
            for value in inputs
            for case in value["cases"]
            if case["pair_randomness"] == randomness
            and case["profile"] == profile
            and case["dimensions"] == 1
        ]
        if not cases:
            continue
        fig, axis = plt.subplots(figsize=(7.6, 4.5), layout="constrained")
        for case in cases:
            points = case["ensemble_trajectory"]
            axis.semilogy(
                [p["time"] for p in points],
                [p["alive_error"]["mean"] for p in points],
                label=f"measured N={case['walkers']}",
                linewidth=1.8,
            )
        points = cases[0]["ensemble_trajectory"]
        field = (
            "noiseless_geometric_envelope"
            if randomness == "shared"
            else "n_independent_geometric_noise_envelope"
        )
        axis.semilogy(
            [p["time"] for p in points],
            [p[field] for p in points],
            color="black",
            linestyle="--",
            linewidth=1.8,
            label="predicted envelope, uniform in N",
        )
        axis.set_xlabel("Physical time")
        axis.set_ylabel("Normalized alive transport squared error")
        axis.set_title(
            "Stored quadratic control: geometric contraction"
            if randomness == "shared"
            else "Stored quadratic control: independent noise and error floor"
        )
        axis.legend(fontsize=9)
        axis.grid(alpha=0.2)
        fig.savefig(destination / name, dpi=180)
        plt.close(fig)


def summarize(report, destination, workspace_root=None):
    """Preserve failures, unavailable inputs and exact expression-level coverage."""
    root = workspace_root or Path(__file__).resolve().parents[1]
    destination.mkdir(parents=True, exist_ok=True)
    if report.get("new_engine_steps") != 0:
        msg = "Stored-run validation must retain an explicit zero fresh-step count"
        raise ValueError(msg)
    source_hashes = {}
    for chapter in report["chapters"]:
        coverage = chapter["coverage"]
        source = root / coverage["source"]
        current = digest(source)
        if current != coverage["source_sha256"]:
            msg = f"Chapter {chapter['chapter']} source changed since its inventory was generated"
            raise ValueError(msg)
        source_hashes[coverage["source"]] = current
    input_hashes = {}
    for original in report["input_reports"]:
        path = resolve_input(root, original)
        input_hashes[original] = digest(path)
    comparisons = []
    missing = []
    unbound = []
    chapter_summaries = []
    for chapter in report["chapters"]:
        number = chapter["chapter"]
        actual_failed = sum(
            check.get("passed") is False or check.get("status") == "violated"
            for check in chapter["comparisons"]
        )
        comparisons.extend({"chapter": number, **check} for check in chapter["comparisons"])
        missing.extend(
            {"chapter": number, **expression}
            for expression in chapter["coverage"]["expression_ledger"]
            if expression["status"] == "not_checked_from_stored_data"
        )
        unbound.extend(
            {"chapter": number, **entry} for entry in chapter["coverage"]["unbound_evidence"]
        )
        chapter_summaries.append({
            "chapter": number,
            "title": chapter.get("title"),
            "comparisons": len(chapter["comparisons"]),
            "comparisons_failed": actual_failed,
            "checked_required_expressions": chapter["coverage"]["checked_required_expressions"],
            "required_expressions": chapter["coverage"]["required_expressions"],
            "missing_required_expressions": chapter["coverage"]["missing_required_expressions"],
            "complete": chapter["coverage"]["complete"] and actual_failed == 0,
            "analysis_summary": chapter.get("summary"),
            "gaps": chapter.get("gaps", []),
        })
    failures = [
        check
        for check in comparisons
        if check.get("passed") is False or check.get("status") == "violated"
    ]
    summary = {
        "new_engine_steps": 0,
        "comparisons": len(comparisons),
        "comparisons_failed": len(failures),
        "unbound_evidence": len(unbound),
        "complete_required_expressions": not missing and not unbound and not failures,
        "chapters": chapter_summaries,
        "source_sha256": source_hashes,
        "input_report_sha256": input_hashes,
    }
    for name, value in (
        ("summary.json", summary),
        ("comparisons.json", comparisons),
        ("missing-estimates.json", missing),
        ("failed-comparisons.json", failures),
        ("unbound-evidence.json", unbound),
    ):
        (destination / name).write_text(json.dumps(value, indent=2) + "\n")
    lines = [
        "# Chapters 4–6: validation from stored runs",
        "",
        (
            f"Fresh engine steps: **0**. Executed comparisons: **{len(comparisons):,}**. "
            f"Rejected/failed comparisons: **{len(failures):,}**."
        ),
        "",
        "| Chapter | Executed comparisons | Failed | Required expressions checked | Required expressions missing |",
        "|---|---:|---:|---:|---:|",
    ]
    for chapter in chapter_summaries:
        lines.append(
            f"| {chapter['chapter']} — {cell(chapter['title'])} | {chapter['comparisons']:,} | "
            f"{chapter['comparisons_failed']:,} | "
            f"{chapter['checked_required_expressions']}/{chapter['required_expressions']} | "
            f"{chapter['missing_required_expressions']} |"
        )
    lines += [
        "",
        (
            "Expression coverage credits a whole quoted source expression only when its owning "
            "label has a passed finite numerical comparison. Source membership, a theorem label, "
            "an empirical endpoint decline or an applicability assertion earns no extra credit. "
            "A finite check retains its input and theorem hypotheses; it is not a universal proof."
        ),
        "",
        (
            "The [complete comparison ledger](comparisons.json) retains measured values, predictions, "
            "relations, scope, hypotheses and source-report locations. "
            "[Missing expressions](missing-estimates.json), "
            "[failed comparisons](failed-comparisons.json) and "
            "[unbound evidence](unbound-evidence.json) remain separate."
        ),
    ]
    for original, chapter in zip(report["chapters"], chapter_summaries):
        lines += ["", f"## Chapter {chapter['chapter']}", ""]
        if original.get("scope"):
            lines.append(str(original["scope"]))
        if chapter["analysis_summary"]:
            lines += [
                "",
                "Analysis counts and applicability:",
                "",
                "```json",
                json.dumps(chapter["analysis_summary"], indent=2),
                "```",
            ]
        lines += ["", "Unavailable inputs and theorem conditions:", ""]
        for gap in chapter["gaps"]:
            lines.append(
                "- " + cell(json.dumps(gap, ensure_ascii=False) if isinstance(gap, dict) else gap)
            )
        if not chapter["gaps"]:
            lines.append(
                "No additional analysis gaps were returned; expression coverage above still applies."
            )
        # Keep every numerical row in the ledger; a compact table previews the families.
        examples = {}
        for check in original["comparisons"]:
            key = check.get("family", check.get("id", "unnamed"))
            examples.setdefault(key, check)
        lines += [
            "",
            "Example comparisons (the linked ledger contains every case):",
            "",
            "| Comparison | Observed | Prediction/bound | Relation | Passed |",
            "|---|---:|---:|---|---|",
        ]
        for check in list(examples.values())[:24]:
            lines.append(
                "| "
                + " | ".join(
                    cell(check.get(key))
                    for key in ("id", "observed", "bound", "relation", "passed")
                )
                + " |"
            )
    lines += [
        "",
        "## Retained input provenance",
        "",
        (
            "Source and input SHA-256 digests are saved in [summary.json](summary.json). "
            "Embedded copies of decay suites do not provide additional independent seeds. "
            "Control contraction, native empirical error, conservative invariant laws and "
            "survival-conditioned QSD laws retain their respective meanings."
        ),
        "",
    ]
    (destination / "summary.md").write_text("\n".join(lines))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    result = summarize(report, args.output)
    if args.plot:
        plot_decay(report, args.output)
        figures = [
            ("quadratic-contraction.png", "Measured decay and predicted contraction envelope"),
            ("quadratic-noise-floor.png", "Independent noise and the predicted error floor"),
        ]
        with (args.output / "summary.md").open("a") as stream:
            stream.write(
                "\n## Predicted and measured decay\n\n"
                "These curves use the certified quadratic controls with cloning disabled, "
                "no velocity cap and no terminal killing. The three noiseless population "
                "curves overlap. Their envelope and the independent-noise envelope are "
                "uniform in N.\n\n"
            )
            for filename, description in figures:
                if (args.output / filename).exists():
                    stream.write(f"![{description}]({filename})\n\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "new_engine_steps",
                    "comparisons",
                    "comparisons_failed",
                    "unbound_evidence",
                    "complete_required_expressions",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
