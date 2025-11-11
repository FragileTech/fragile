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
import json
import logging
from pathlib import Path
import re

import click
import flogging

from mathster.iterators import discover_chapter_markdown_files, discover_registry_folders


flogging.setup(level="INFO", allow_trailing_dot=True)

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}

CHAPTER_FILENAME_PATTERN = re.compile(r"^\d{2}_.+\.md$")
DEFAULT_CHAPTER_ROOTS: tuple[Path, ...] = (Path("docs/source"), Path("old_docs/source"))


def _normalize_document_argument(value: str | Path | None) -> str:
    """Convert CLI document inputs (id/path/.md) into a document_id string."""

    if value is None:
        return ""

    text = str(value).strip()
    if not text:
        return ""

    candidate = Path(text)
    suffix = candidate.suffix.lower()
    if suffix == ".md":
        return candidate.stem
    if suffix:
        return candidate.stem

    return candidate.name or text


def _locate_markdown_file(document: str | Path | None) -> Path | None:
    """Best-effort resolution of a document argument into a markdown file."""

    if document is None:
        return None

    text = str(document).strip()
    if not text:
        return None

    raw = Path(text).expanduser()
    candidates: list[Path] = []

    def _add_candidate(path: Path) -> None:
        path = path.expanduser()
        if path not in candidates:
            candidates.append(path)

    _add_candidate(raw)
    if raw.suffix != ".md":
        _add_candidate(raw.with_suffix(".md"))

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    if raw.is_absolute():
        return None

    rel_variations = [raw]
    if raw.suffix != ".md":
        rel_variations.append(raw.with_suffix(".md"))

    stem_name = raw.stem if raw.suffix else raw.name

    for root in DEFAULT_CHAPTER_ROOTS:
        if not root.exists():
            continue
        for rel in rel_variations:
            _add_candidate(root / rel)
        if stem_name:
            _add_candidate(root / stem_name)
            _add_candidate(root / f"{stem_name}.md")

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    if stem_name:
        for root in DEFAULT_CHAPTER_ROOTS:
            if not root.exists():
                continue
            match = next(root.rglob(f"{stem_name}.md"), None)
            if match:
                return match

    return None


def _run_parsing_stage(markdown_file: Path) -> dict:
    """Execute the directive parsing stage for ``markdown_file``."""
    from mathster.directives.command import run_directive_extraction

    return run_directive_extraction(
        markdown_file=markdown_file,
        output_dir=None,
        preview=False,
        validate=True,
    )


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
    from mathster.agents.run_extraction import AGENT_CONFIG, run_agents_in_sequence

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(levelname)s - %(message)s",
    )

    source_list = list(sources)
    if not source_list:
        source_list = discover_chapter_markdown_files()
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
        source_list = discover_chapter_markdown_files()
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
    "preprocess",
    help=(
        "Run the preprocess-extraction stage (merging directives + extracts) for the "
        "selected documents or, by default, every document with extraction data."
    ),
)
@click.argument("documents", nargs=-1)
@click.option(
    "--log-level",
    type=str,
    default="INFO",
    show_default=True,
    help="Logging level for preprocess stage execution.",
)
def preprocess_command(documents: tuple[str, ...], log_level: str) -> None:
    """Invoke mathster.preprocess_extraction.process_stage.run_all_stages."""
    from mathster.preprocess_extraction.process_stage import run_all_stages

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(levelname)s - %(message)s",
    )

    if documents:
        targets: list[str | Path] = list(documents)
    else:
        registry_dirs = discover_registry_folders()
        if not registry_dirs:
            msg = "No document registries with extraction data were found."
            raise click.ClickException(msg)
        targets = list(dict.fromkeys(registry_dir.parent for registry_dir in registry_dirs))
        click.echo(f"Discovered {len(targets)} document registries with extraction data.")

    failures: list[tuple[str, str]] = []
    for idx, target in enumerate(targets, start=1):
        label = str(target)
        click.echo(f"[{idx}/{len(targets)}] Running preprocess stages for {label}...")
        try:
            outputs = run_all_stages(target)
        except Exception as exc:  # pragma: no cover - CLI surface
            message = str(exc)
            failures.append((label, message))
            click.echo(f"  × {message}")
            continue

        for output_path in outputs:
            click.echo(f"  → {output_path}")

    if failures:
        summary = "\n".join(f"  - {label}: {message}" for label, message in failures)
        raise click.ClickException(
            f"{len(failures)} document(s) failed during preprocessing:\n{summary}"
        )


@cli.command(
    "connectivity",
    help=(
        "Refresh directives via parsing and generate a markdown connectivity report "
        "for a document (uses preprocess registries when available, otherwise falls "
        "back to directives)."
    ),
)
@click.argument("document", type=str)
@click.option(
    "--preprocess-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path, exists=True),
    default=None,
    help="Explicit preprocess registry directory; overrides auto-discovery.",
)
def connectivity_command(document: str, preprocess_dir: Path | None) -> None:
    """CLI entry point for document connectivity reports."""
    from mathster.reports.document_connectivity_report import (
        generate_document_connectivity_report,
    )

    markdown_file = _locate_markdown_file(document)
    if markdown_file is None:
        raise click.ClickException(
            f"Unable to locate markdown file for '{document}'. Provide a valid .md path or document id."
        )

    document_id = _normalize_document_argument(document)
    if not document_id:
        msg = "Document identifier must be a non-empty string."
        raise click.ClickException(msg)

    if preprocess_dir is not None:
        preprocess_dir = preprocess_dir.expanduser()

    click.echo(f"Parsing directives for {markdown_file} to refresh connectivity data...")
    try:
        _run_parsing_stage(markdown_file)
    except Exception as exc:  # pragma: no cover - CLI surface
        raise click.ClickException(f"Parsing stage failed for {markdown_file}: {exc}") from exc

    doc_workspace = markdown_file.parent / markdown_file.stem
    doc_workspace.mkdir(parents=True, exist_ok=True)

    click.echo(f"Building registry/directives for {doc_workspace}...")
    try:
        from mathster.registry.directives_stage import process_document

        process_document(doc_workspace, force=True, verbose=False)
    except Exception as exc:  # pragma: no cover - CLI surface
        raise click.ClickException(
            f"Directive registry build failed for {doc_workspace}: {exc}"
        ) from exc

    try:
        report = generate_document_connectivity_report(
            document_id=document_id,
            preprocess_dir=preprocess_dir,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(report)


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
    from mathster.registry.extract_stage import (
        build_unified_extract_registry,
        build_unified_preprocess_registry,
    )

    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(message)s",
    )

    target_path = path
    if not target_path.exists():
        raise click.ClickException(f"Path does not exist: {target_path}")

    any_success = False

    try:
        if build_unified_only:
            success = build_unified_registry(
                target_path,
                output_dir=unified_output,
                verbose=verbose,
            )
            any_success = success
        elif batch:
            results = process_batch(
                target_path,
                force=force,
                verbose=verbose,
                build_unified=not no_unified,
                unified_output=unified_output,
            )
            any_success = any(results.values())
            all(results.values())
            failed_docs = [doc for doc, ok in results.items() if not ok]
            if failed_docs:
                click.echo(
                    "Some documents failed during directives-stage processing: "
                    + ", ".join(sorted(failed_docs))
                )
            success = any_success
        else:
            success = process_document(target_path, force=force, verbose=verbose)
            any_success = success

        if any_success and not no_unified:
            # Build corpus-level extract registry after directives succeed.
            build_unified_extract_registry(
                target_path,
                output_dir=unified_output,
                verbose=verbose,
            )
            build_unified_preprocess_registry(
                target_path,
                output_dir=unified_output,
                verbose=verbose,
            )
    except Exception as exc:  # pragma: no cover - CLI surface
        raise click.ClickException(str(exc)) from exc

    if not success:
        msg = "Directive registry build failed"
        raise click.ClickException(msg)


@cli.command(
    "search",
    help="Lookup a registry entity by label and print its JSON payload.",
)
@click.argument("label", type=str)
@click.option(
    "--stage",
    type=click.Choice(["auto", "preprocess", "directives"]),
    default="auto",
    show_default=True,
    help="Registry stage to search (auto tries preprocess first, then directives).",
)
@click.option(
    "--preprocess-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Custom preprocess registry directory (defaults to unified_registry/preprocess).",
)
@click.option(
    "--directives-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Custom directives registry directory (defaults to unified_registry/directives).",
)
def search_command(
    label: str,
    stage: str,
    preprocess_dir: Path | None,
    directives_dir: Path | None,
) -> None:
    """Load registry JSON for ``label`` and stream it to stdout."""
    from mathster.registry import search as registry_search

    label = label.strip()
    if not label:
        msg = "Label must be a non-empty string."
        raise click.ClickException(msg)

    selected_stage = stage or "auto"
    registry_source = None
    entity: dict | None = None

    if selected_stage in {"auto", "preprocess"}:
        entity = registry_search.get_preprocess_label(label, preprocess_dir=preprocess_dir)
        if entity is not None:
            registry_source = "preprocess"

    if entity is None and selected_stage in {"auto", "directives"}:
        entity = registry_search.get_directive_label(label, directives_dir=directives_dir)
        if entity is not None:
            registry_source = "directives"

    if entity is None:
        if selected_stage == "preprocess":
            scope = "the preprocess registry"
        elif selected_stage == "directives":
            scope = "the directives registry"
        else:
            scope = "either registry"
        raise click.ClickException(f"Label '{label}' was not found in {scope}.")

    assert registry_source is not None  # For type checkers
    click.echo(f"[{registry_source} registry] {label}")
    click.echo(json.dumps(entity, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    cli()
