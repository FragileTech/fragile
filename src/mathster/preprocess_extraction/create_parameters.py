from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict

from pydantic import ValidationError

from mathster.preprocess_extraction.data_models import Parameter
from mathster.preprocess_extraction.utils import (
    load_json,
    resolve_document_directory,
)


logger = logging.getLogger(__name__)


@dataclass
class ParameterOccurrence:
    """Single appearance of a parameter inside a preprocess entity."""

    parameter: Parameter
    source_file: str
    entity_label: str | None
    entity_type: str | None
    start_line: int | None


def _normalize_symbol(symbol: str | None) -> str | None:
    if not symbol:
        return None
    normalized = symbol.strip()
    return normalized or None


def _normalize_str(value: object) -> str | None:
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return None


def _build_label(name: str | None, symbol: str) -> str:
    base = name or symbol or "parameter"
    normalized = base.lower().replace(" ", "-")
    normalized = normalized.strip("-") or "parameter"
    return f"obj-{normalized}"


def _extract_start_line(entry: dict) -> int | None:
    span = entry.get("span")
    if isinstance(span, dict):
        start = span.get("start_line")
        if isinstance(start, int):
            return start
        if isinstance(start, str) and start.isdigit():
            return int(start)
    raw_start = entry.get("start_line")
    if isinstance(raw_start, int):
        return raw_start
    if isinstance(raw_start, str) and raw_start.isdigit():
        return int(raw_start)
    return None


def _iter_preprocess_files(preprocess_dir: Path) -> Iterable[Path]:
    for path in sorted(preprocess_dir.glob("*.json")):
        if path.name == "parameter.json":
            continue
        yield path


def _collect_parameters(preprocess_dir: Path) -> list[ParameterOccurrence]:
    """Return parameters paired with their source file/label."""

    collected: list[ParameterOccurrence] = []

    for json_path in _iter_preprocess_files(preprocess_dir):
        try:
            payload = load_json(json_path)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to load %s: %s", json_path, exc)
            continue

        if not isinstance(payload, list):
            logger.debug("Skipping %s (expected list payload).", json_path)
            continue

        for entry in payload:
            if not isinstance(entry, dict):
                continue
            label = entry.get("label")
            candidates = entry.get("parameters")
            if not isinstance(candidates, list) or not candidates:
                continue

            entity_type = _normalize_str(entry.get("type"))
            start_line = _extract_start_line(entry)

            for raw_param in candidates:
                if not isinstance(raw_param, dict):
                    continue
                try:
                    parameter = Parameter(**raw_param)
                except ValidationError as exc:
                    logger.warning(
                        "Skipping invalid parameter in %s (%s): %s",
                        json_path.name,
                        label or "<unknown>",
                        exc,
                    )
                    continue
                occurrence = ParameterOccurrence(
                    parameter=parameter,
                    source_file=json_path.name,
                    entity_label=label,
                    entity_type=entity_type,
                    start_line=start_line,
                )
                collected.append(occurrence)

    return collected


def _deduplicate_parameters(
    occurrences: Iterable[ParameterOccurrence],
) -> tuple[list[dict], Dict[str, list[ParameterOccurrence]]]:
    """Merge parameters by their LaTeX/symbol representation."""

    merged: dict[str, dict] = {}
    occ_map: Dict[str, list[ParameterOccurrence]] = {}

    for occurrence in occurrences:
        symbol = _normalize_symbol(occurrence.parameter.symbol)
        if not symbol:
            logger.warning(
                "Encountered parameter without a symbol (source: %s %s); skipping.",
                occurrence.source_file,
                occurrence.entity_label or "<unknown>",
            )
            continue

        occ_map.setdefault(symbol, []).append(occurrence)

        entry = merged.setdefault(
            symbol,
            {
                "symbol": symbol,
                "name": None,
                "description": None,
                "constraints": set(),
                "tags": set(),
            },
        )

        parameter = occurrence.parameter

        if not entry["name"] and parameter.name:
            entry["name"] = parameter.name
        if not entry["description"] and parameter.description:
            entry["description"] = parameter.description

        entry["constraints"].update(
            constraint.strip() for constraint in parameter.constraints if constraint and constraint.strip()
        )
        entry["tags"].update(tag.strip() for tag in parameter.tags if tag and tag.strip())
    normalized: list[dict] = []
    for symbol in sorted(merged):
        payload = merged[symbol]
        occ_list = sorted(
            occ_map.get(symbol, []),
            key=lambda item: (
                item.start_line if item.start_line is not None else 10**9,
                item.entity_label or "",
            ),
        )
        defined_label = occ_list[0].entity_label if occ_list and occ_list[0].entity_label else None
        reference_labels = sorted(
            {occ.entity_label for occ in occ_list[1:] if occ.entity_label}
        )
        normalized.append(
            {
                "label": _build_label(payload["name"], symbol),
                "symbol": symbol,
                "name": payload["name"],
                "description": payload["description"],
                "constraints": sorted(payload["constraints"]),
                "tags": sorted(payload["tags"]),
                "defined_in": defined_label,
                "references": reference_labels,
            }
        )

    return normalized, occ_map


def _format_occurrence(label: str | None, entity_type: str | None, source_file: str, start_line: int | None) -> str:
    label_str = label or "<unknown>"
    type_str = entity_type or "entry"
    line_str = f"line {start_line}" if start_line is not None else "line ?"
    return f"{type_str} `{label_str}` ({line_str}; {source_file})"


def generate_parameter_markdown_report(
    document_id: str,
    parameters: list[dict],
    occurrences: Dict[str, list[ParameterOccurrence]],
) -> str:
    """Return a markdown report summarizing parameter usage."""

    total_mentions = sum(len(items) for items in occurrences.values())
    lines: list[str] = [
        f"# Parameter Usage Report: `{document_id}`",
        "",
        f"- **Total unique parameters:** {len(parameters)}",
        f"- **Total mentions across directives:** {total_mentions}",
    ]

    for entry in parameters:
        symbol = entry["symbol"]
        occ_list = sorted(
            occurrences.get(symbol, []),
            key=lambda item: (
                item.start_line if item.start_line is not None else 10**9,
                item.entity_label or "",
            ),
        )
        lines.extend(("", f"## `{symbol}` — {entry.get('name') or '(unnamed parameter)' }"))
        description = entry.get("description") or "(no description available)"
        lines.append(f"- **Description:** {description}")

        if not occ_list:
            lines.append("- **Introduced in:** (no directive data)")
            lines.append("- **Referenced in:** (no directive data)")
            continue

        intro = occ_list[0]
        lines.append(f"- **Introduced in:** {_format_occurrence(intro.entity_label, intro.entity_type, intro.source_file, intro.start_line)}")

        references = occ_list[1:]
        if references:
            lines.append("- **Referenced in:**")
            for ref in references:
                lines.append(
                    f"  - {_format_occurrence(ref.entity_label, ref.entity_type, ref.source_file, ref.start_line)}"
                )
        else:
            lines.append("- **Referenced in:** (no additional directives)")

    return "\n".join(lines).strip() + "\n"


def create_parameter_registry(
    document: str | Path,
    *,
    output_path: Path | None = None,
) -> tuple[Path, list[dict]]:
    """Build registry/preprocess/parameter.json for a document."""

    document_dir = resolve_document_directory(document)
    registry_dir = document_dir / "registry"
    if not registry_dir.exists():
        raise FileNotFoundError(f"No registry directory found under {document_dir}")

    preprocess_dir = registry_dir / "preprocess"
    if not preprocess_dir.exists():
        raise FileNotFoundError(
            f"No preprocess directory found under {registry_dir}. "
            "Run preprocess stages before creating parameters."
        )

    collected = _collect_parameters(preprocess_dir)
    unified, occurrence_map = _deduplicate_parameters(collected)

    destination = output_path or (preprocess_dir / "parameter.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(unified, indent=2), encoding="utf-8")
    logger.info(
        "Wrote %s parameter(s) to %s",
        len(unified),
        destination,
    )
    report = generate_parameter_markdown_report(document_dir.name, unified, occurrence_map)
    return destination, unified, report


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate parameters from preprocess JSON files and write "
            "registry/preprocess/parameter.json for a single document."
        )
    )
    parser.add_argument(
        "document",
        help=(
            "Document identifier (directory, markdown file, or short name). "
            "Examples: 'docs/source/1_euclidean_gas/03_cloning', '03_cloning.md', '03_cloning'."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional custom destination for the generated parameter.json file.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: INFO).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s - %(message)s",
    )

    try:
        destination, unified, report = create_parameter_registry(args.document, output_path=args.output)
    except Exception as exc:  # pragma: no cover - CLI surface
        logger.error("%s", exc)
        return 1

    print(destination)
    print(json.dumps(unified, indent=2))
    print("")
    print(report)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
