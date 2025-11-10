from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Protocol, Sequence

from mathster.preprocess_extraction import (
    process_algorithms,
    process_assumptions,
    process_axioms,
    process_corollaries,
    process_definitions,
    process_lemmas,
    process_proofs,
    process_propositions,
    process_remarks,
    process_theorems,
)
from mathster.preprocess_extraction.utils import resolve_document_directory


logger = logging.getLogger(__name__)


class StageRunner(Protocol):  # pragma: no cover - typing helper
    def __call__(self, document: str | Path, *, output_path: Path | None = None) -> Path: ...


@dataclass(frozen=True)
class StageSpec:
    name: str
    runner: StageRunner
    output_filename: str


STAGES: tuple[StageSpec, ...] = (
    StageSpec("proof", process_proofs.preprocess_document_proofs, "proof.json"),
    StageSpec(
        "definition", process_definitions.preprocess_document_definitions, "definition.json"
    ),
    StageSpec(
        "assumption", process_assumptions.preprocess_document_assumptions, "assumption.json"
    ),
    StageSpec("axiom", process_axioms.preprocess_document_axioms, "axiom.json"),
    StageSpec("lemma", process_lemmas.preprocess_document_lemmas, "lemma.json"),
    StageSpec(
        "proposition", process_propositions.preprocess_document_propositions, "proposition.json"
    ),
    StageSpec("theorem", process_theorems.preprocess_document_theorems, "theorem.json"),
    StageSpec("corollary", process_corollaries.preprocess_document_corollaries, "corollary.json"),
    StageSpec("remark", process_remarks.preprocess_document_remarks, "remark.json"),
    StageSpec("algorithm", process_algorithms.preprocess_document_algorithms, "algorithm.json"),
)


def run_all_stages(
    document: str | Path, stage_specs: Sequence[StageSpec] | None = None
) -> list[Path]:
    document_dir = resolve_document_directory(document)
    registry_dir = document_dir / "registry"
    if not registry_dir.exists():
        raise FileNotFoundError(f"No registry directory found under {document_dir}")

    preprocess_dir = registry_dir / "preprocess"
    preprocess_dir.mkdir(parents=True, exist_ok=True)

    specs = list(stage_specs or STAGES)
    results: list[Path] = []
    for spec in specs:
        destination = preprocess_dir / spec.output_filename
        logger.info("Running %s stage -> %s", spec.name, destination)
        try:
            result_path = spec.runner(document_dir, output_path=destination)
            results.append(result_path)
        except FileNotFoundError as exc:
            logger.warning("Skipping %s stage: %s", spec.name, exc)
            continue

    return results


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run every preprocess extraction stage sequentially for a document registry.",
    )
    parser.add_argument(
        "document",
        help=(
            "Document identifier (directory, markdown file, or document name). "
            "Examples: 'docs/source/1_euclidean_gas/03_cloning', "
            "'docs/source/1_euclidean_gas/03_cloning.md', '03_cloning'."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: INFO).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s - %(message)s",
    )

    try:
        outputs = run_all_stages(args.document)
    except Exception as exc:  # pragma: no cover - CLI entry point
        logger.error("%s", exc)
        return 1

    for path in outputs:
        print(path)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
