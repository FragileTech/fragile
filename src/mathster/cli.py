"""Unified command-line interface for Mathster tooling.

This module exposes a Click-based CLI that consolidates the most common
operations used throughout the mathematical document pipeline:

- ``mathster extract``: Run DSPy-based directive extraction agents via
  ``agents/run_extraction.py``.
- ``mathster parse``: Directive hint extraction (structural parsing only).
- ``mathster validate``: Schema/relationship/framework validation helpers.

Additional subcommands can be layered on top of this entry point without
changing the ``pyproject`` script definition (``mathster = mathster.cli:cli``).
"""

from __future__ import annotations

import importlib.metadata
import logging
from pathlib import Path
import re
from typing import Sequence

import click
import flogging

from mathster.agents.run_extraction import AGENT_CONFIG, run_agents_in_sequence


flogging.setup(level="INFO")

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}

CHAPTER_FILENAME_PATTERN = re.compile(r"^\d{2}_.+\.md$")
DEFAULT_CHAPTER_ROOTS: tuple[Path, ...] = (Path("docs/source"), Path("old_docs/source"))


def _resolve_version() -> str:
    """Return the installed fragile package version (best effort)."""
    try:
        return importlib.metadata.version("fragile")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover - local dev fallback
        return "0.0.dev0"


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(version=_resolve_version(), message="%(prog)s %(version)s")
def cli() -> None:
    """Mathster toolkit CLI."""


def _discover_chapter_markdown_files(roots: Sequence[Path] | None = None) -> list[Path]:
    """Return all chapter markdown files (pattern ``NN_chapter_name.md``)."""
    roots = roots or DEFAULT_CHAPTER_ROOTS
    matches: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.md"):
            if CHAPTER_FILENAME_PATTERN.match(path.name):
                matches.append(path)
    return sorted(set(matches))


@cli.command(
    "extract",
    help="Run the Mathster extraction agents for one or more documents.",
)
@click.argument(
    "sources",
    nargs=-1,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--agent",
    "agents",
    multiple=True,
    help="Subset of agents to run (repeat option for multiple; defaults to all).",
)
@click.option(
    "--lm",
    type=str,
    default="xai/grok-4-fast-reasoning-latest",
    show_default=True,
    help="LM spec forwarded to run_extraction.py.",
)
@click.option(
    "--passes",
    type=int,
    default=5,
    show_default=True,
    help="Number of DSPy Refine passes per agent.",
)
@click.option(
    "--threshold",
    type=float,
    default=None,
    help="Override reward threshold for all agents.",
)
@click.option(
    "--max-tokens",
    type=int,
    default=None,
    help="Override max tokens for all agents.",
)
@click.option(
    "--log-level",
    type=str,
    default="INFO",
    show_default=True,
    help="Logging level for extraction agents.",
)
def extract_command(
    sources: tuple[Path, ...],
    agents: tuple[str, ...],
    lm: str,
    passes: int,
    threshold: float | None,
    max_tokens: int | None,
    log_level: str,
) -> None:
    """Invoke the DSPy extraction orchestrator directly."""

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(levelname)s - %(message)s",
    )

    source_list = list(sources)
    if not source_list:
        source_list = _discover_chapter_markdown_files()
        if not source_list:
            msg = "No chapter markdown files were found (pattern NN_chapter-name.md)."
            raise click.ClickException(msg)
        click.echo(
            f"Discovered {len(source_list)} chapter markdown files under "
            f"{', '.join(str(root) for root in DEFAULT_CHAPTER_ROOTS if root.exists())}."
        )

    agent_list = list(agents) if agents else list(AGENT_CONFIG.keys())

    for idx, source in enumerate(source_list, start=1):
        click.echo(f"[{idx}/{len(source_list)}] Running extraction agents for {source}...")

        try:
            run_agents_in_sequence(
                doc_path=str(source),
                agent_names=agent_list,
                lm_spec=lm,
                passes=passes,
                threshold_override=threshold,
                max_tokens_override=max_tokens,
            )
        except Exception as exc:  # pragma: no cover - CLI surface
            raise click.ClickException(f"{source}: {exc}") from exc


@cli.command(
    "parse",
    help=(
        "Run the directive extractor (structure-only parsing) and write chapter-level "
        "JSON files under <doc_dir>/<stem>/directives/."
    ),
)
@click.argument(
    "markdown_files",
    nargs=-1,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Custom document workspace (defaults to <doc_dir>/<stem>).",
)
@click.option(
    "--no-preview",
    is_flag=True,
    help="Disable console previews of extracted directives.",
)
@click.option(
    "--no-validate",
    is_flag=True,
    help="Skip structural validation checks.",
)
def parse_command(
    markdown_files: tuple[Path, ...],
    output_dir: Path | None,
    no_preview: bool,
    no_validate: bool,
) -> None:
    """Invoke the directive extractor instead of the DSPy orchestrator."""
    from mathster.directives.command import run_directive_extraction

    source_list = list(markdown_files)
    if not source_list:
        source_list = _discover_chapter_markdown_files()
        if not source_list:
            msg = "No chapter markdown files were found (pattern NN_chapter-name.md)."
            raise click.ClickException(msg)
        click.echo(
            f"Discovered {len(source_list)} chapter markdown files under "
            f"{', '.join(str(root) for root in DEFAULT_CHAPTER_ROOTS if root.exists())}."
        )

    if output_dir is not None and len(source_list) != 1:
        msg = "--output-dir can only be used with a single document."
        raise click.ClickException(msg)

    for idx, markdown_file in enumerate(source_list, start=1):
        click.echo(f"[{idx}/{len(source_list)}] Parsing directives for {markdown_file}...")
        try:
            result = run_directive_extraction(
                markdown_file=markdown_file,
                output_dir=output_dir,
                preview=not no_preview,
                validate=not no_validate,
            )
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc

        if not no_preview:
            for chapter in result["chapters"]:
                preview_text = chapter.get("preview")
                if preview_text:
                    click.echo(preview_text)
                    click.echo("")

        click.echo(
            f"  → Wrote {len(result['chapters'])} directive files to {result['output_dir']} "
            f"({result['total_directives']} directives total)."
        )


@cli.command(
    "registry",
    help=(
        "Build directive registries for one document or batches (defaults to docs/source, "
        "batch mode, forced rebuild, verbose output)."
    ),
)
@click.argument(
    "path",
    required=False,
    default="docs/source",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
)
@click.option(
    "--batch/--no-batch",
    default=True,
    help="Process entire directory tree when enabled.",
)
@click.option(
    "--build-unified-only",
    is_flag=True,
    help="Only assemble the unified registry from existing document registries.",
)
@click.option(
    "--force/--no-force",
    default=True,
    help="Force regeneration even when registry files already exist.",
)
@click.option(
    "--no-unified",
    is_flag=True,
    help="Skip unified registry build during batch processing.",
)
@click.option(
    "--unified-output",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Custom destination for unified registry files.",
)
@click.option(
    "--verbose/--no-verbose",
    default=True,
    help="Toggle verbose logging for directives-stage processing.",
)
def registry_command(
    path: Path,
    batch: bool,
    build_unified_only: bool,
    force: bool,
    no_unified: bool,
    unified_output: Path | None,
    verbose: bool,
) -> None:
    """Expose directives-stage registry building through the CLI."""
    from mathster.registry.directives_stage import (
        build_unified_registry,
        process_batch,
        process_document,
    )
    from mathster.registry.extract_stage import build_unified_extract_registry

    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(message)s",
    )

    target_path = path
    if not target_path.exists():
        raise click.ClickException(f"Path does not exist: {target_path}")

    try:
        if build_unified_only:
            success = build_unified_registry(
                target_path,
                output_dir=unified_output,
                verbose=verbose,
            )
        elif batch:
            results = process_batch(
                target_path,
                force=force,
                verbose=verbose,
                build_unified=not no_unified,
                unified_output=unified_output,
            )
            success = all(results.values())
        else:
            success = process_document(target_path, force=force, verbose=verbose)

        if success and not no_unified:
            # Build corpus-level extract registry after directives succeed.
            build_unified_extract_registry(
                target_path,
                output_dir=unified_output,
                verbose=verbose,
            )
    except Exception as exc:  # pragma: no cover - CLI surface
        raise click.ClickException(str(exc)) from exc

    if not success:
        msg = "Directive registry build failed"
        raise click.ClickException(msg)


@cli.command("validate", help="Validate refined or pipeline entities.")
@click.option(
    "--refined-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Directory containing refined_data/ or pipeline_data/ exports.",
)
@click.option(
    "--mode",
    type=click.Choice(["schema", "relationships", "framework", "complete"]),
    default="schema",
    show_default=True,
    help="Validation scope.",
)
@click.option(
    "-e",
    "--entity-type",
    "entity_types",
    type=click.Choice([
        "theorems",
        "axioms",
        "objects",
        "parameters",
        "mathster",
        "remarks",
        "equations",
    ]),
    multiple=True,
    help="Limit validation to selected entity types.",
)
@click.option(
    "--output-report",
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    help="Optional markdown summary output path.",
)
@click.option(
    "--output-json",
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    help="Optional JSON summary output path.",
)
@click.option("--strict", is_flag=True, help="Treat warnings as errors.")
@click.option(
    "--glossary",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("docs/glossary.md"),
    show_default=True,
    help="Glossary file used for framework validation.",
)
@click.option(
    "--validation-mode",
    type=click.Choice(["refined", "pipeline"]),
    default="refined",
    show_default=True,
    help="Schema validation mode (refined_data vs. pipeline_data).",
)
def validate_command(
    refined_dir: Path,
    mode: str,
    entity_types: tuple[str, ...],
    output_report: Path | None,
    output_json: Path | None,
    strict: bool,
    glossary: Path,
    validation_mode: str,
) -> None:
    """Invoke validation helpers from mathster.tools.validation."""
    from mathster.tools.validation.cli import (
        validate_complete,
        validate_framework,
        validate_relationships,
        validate_schema,
    )
    from mathster.tools.validation.validation_report import ValidationReport

    selected_types = list(entity_types) or [
        "theorems",
        "axioms",
        "objects",
        "parameters",
        "mathster",
        "remarks",
        "equations",
    ]

    click.echo(f"Validation mode: {mode}")
    click.echo(f"Entity types: {', '.join(selected_types)}")
    click.echo(f"Source directory: {refined_dir}")
    click.echo("")

    if mode == "schema":
        result = validate_schema(refined_dir, selected_types, strict, validation_mode)
    elif mode == "relationships":
        result = validate_relationships(refined_dir, selected_types, strict)
    elif mode == "framework":
        result = validate_framework(refined_dir, selected_types, strict, glossary)
    elif mode == "complete":
        result = validate_complete(refined_dir, selected_types, strict, glossary)
    else:  # pragma: no cover - safeguarded by click.Choice
        raise click.ClickException(f"Unknown validation mode: {mode}")

    report = ValidationReport(result, refined_dir, mode)
    report.print_summary()

    if output_report:
        report.save_markdown(output_report)
        click.echo(f"Wrote markdown report to {output_report}")

    if output_json:
        report.save_json(output_json)
        click.echo(f"Wrote JSON report to {output_json}")

    if not result.is_valid:
        msg = "Validation failed"
        raise click.ClickException(msg)


if __name__ == "__main__":  # pragma: no cover
    cli()
