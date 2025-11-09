#!/usr/bin/env python3
"""
Shared infrastructure for Mathster extraction agents.

The extraction agents under ``src/mathster/agents`` currently duplicate a fair
amount of plumbing: locating registry files, stripping line numbers, clamping
context windows, reading JSON/JSONL batches, and emitting result files.  This
module centralizes those patterns so that future agents (and future refactors of
the existing theorem/definition/proof agents) can import the same utilities.

The helpers defined here are intentionally conservative:

* They make no assumptions about the downstream DSPy signature being used.
* They expose small, composable primitives (path helpers, parsing helpers,
  context window utilities, JSON safety helpers, etc.).
* They avoid rewriting the current agents; instead, the goal is to provide
  ready-to-use building blocks for a subsequent refactor.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Iterator, Sequence

from tqdm import tqdm

from mathster.agents.signatures import (
    ExtractSignature,
    ExtractWithParametersSignature,
    ImplicitReference,
    Parameter,
    to_jsonable,
)


logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_CONTEXT_WINDOW",
    "LATEX_FENCE_PATTERN",
    "LINE_NUMBER_PATTERN",
    "METADATA_PATTERN",
    "URI_PATTERN",
    "DirectiveAgentPaths",
    "DirectiveInputBuilder",
    "ExtractSignature",
    "ExtractWithParametersSignature",
    "clamp_text_window",
    "default_input_builder",
    "directive_text_from_object",
    "extract_context_window",
    "fence_free",
    "has_duplicates",
    "iter_registry_items",
    "nonempty",
    "reasonable_text_len",
    "run_directive_extraction_loop",
    "safe_json_dumps",
    "safe_json_loads",
    "to_jsonable",
    "strip_line_numbers",
]


# --------------------------------------------------------------------------------------
# Regular expressions & constants shared across reward functions and text utilities
# --------------------------------------------------------------------------------------


URI_PATTERN = re.compile(r"(https?://|file://|s3://|gs://)", re.IGNORECASE)
METADATA_PATTERN = re.compile(r"\b(line|page|timestamp|uuid|sha256)\b", re.IGNORECASE)
LINE_NUMBER_PATTERN = re.compile(r"^\s*\d+:\s?")
LATEX_FENCE_PATTERN = re.compile(r"(\$\$|\\\[|\\\]|\\begin\{equation\}|\\end\{equation\}|'')")

DEFAULT_CONTEXT_WINDOW = 320


# --------------------------------------------------------------------------------------
# Path plumbing
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DirectiveAgentPaths:
    """
    Convenience container for the file structure each directive agent uses.

    Attributes
    ----------
    document_path:
        Path to the numbered Markdown document (e.g., ``07_mean_field.md``).
    registry_path:
        Path to the registry ``*.json`` (e.g., ``registry/directives/theorem.json``).
    extract_path:
        Output destination (e.g., ``registry/extract/theorem.json``).
    """

    document_path: Path
    registry_path: Path
    extract_path: Path

    @classmethod
    def build(
        cls,
        document_path: str | Path,
        directive_basename: str,
        *,
        output_basename: str | None = None,
    ) -> DirectiveAgentPaths:
        """
        Factory that mirrors the current folder layout used by the agents.

        Parameters
        ----------
        document_path:
            Path to the source markdown file.
        directive_basename:
            File stem inside ``registry/directives`` (e.g., ``theorem`` or
            ``lemma``). The ``.json`` extension is added automatically.
        output_basename:
            Optional override for the output file name (defaults to the same as
            ``directive_basename``).
        """

        doc_path = Path(document_path)
        doc_folder = doc_path.parent / doc_path.stem
        registry = doc_folder / "registry" / "directives" / f"{directive_basename}.json"
        extract_dir = doc_folder / "registry" / "extract"
        output_name = output_basename or directive_basename
        extract = extract_dir / f"{output_name}.json"
        return cls(document_path=doc_path, registry_path=registry, extract_path=extract)

    @property
    def document_folder(self) -> Path:
        """Directory that contains per-document artifacts."""
        return self.document_path.parent / self.document_path.stem

    def ensure_output_directory(self) -> None:
        """Create the extract directory if it does not exist."""
        self.extract_path.parent.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------------------
# JSON utilities
# --------------------------------------------------------------------------------------


def iter_registry_items(registry_path: Path) -> Iterator[dict[str, Any]]:
    """
    Yield directive objects from a registry JSON/JSONL file.

    The registry emits either:
        * a JSON document containing ``{"items": [...]}``
        * a bare JSON list (``[{}, {}, ...]``)
        * or a JSONL file where each line is an object.
    """

    if not registry_path.exists():
        raise FileNotFoundError(f"Registry file not found: {registry_path}")

    text = registry_path.read_text(encoding="utf-8")
    if registry_path.suffix.lower() == ".jsonl":
        for idx, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Failed to parse line %s in %s: %s", idx, registry_path, exc)
                continue
            if isinstance(obj, dict):
                yield obj
            else:
                logger.debug("Skipping non-object entry on line %s of %s", idx, registry_path)
    else:
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Could not parse registry JSON: {registry_path}") from exc

        items: Iterable[Any]
        if isinstance(raw, dict) and "items" in raw:
            items = raw["items"]  # type: ignore[assignment]
        else:
            items = raw

        if not isinstance(items, Iterable):
            raise TypeError(
                f"Registry JSON is not iterable (expected list or items): {registry_path}"
            )

        for item in items:
            if isinstance(item, dict):
                yield item
            else:
                logger.debug("Skipping non-object entry in %s: %r", registry_path, item)


def safe_json_loads(payload: str | None, default: Any):
    """
    Defensive ``json.loads`` wrapper.

    Returns ``default`` when the payload is empty or fails to parse. The current
    agents often use this in reward functions to avoid exceptions during DSPy
    refinement.
    """

    if not payload or not payload.strip():
        return default
    try:
        return json.loads(payload)
    except Exception:
        return default


def safe_json_dumps(data: Any, *, indent: int = 2) -> str:
    """
    Dump ``data`` to JSON using UTF-8 friendly settings.

    This mirrors ``json.dumps(..., ensure_ascii=False, indent=2)`` so agents can
    emit files via a single helper call.
    """

    return json.dumps(data, ensure_ascii=False, indent=indent)


# --------------------------------------------------------------------------------------
# Text helpers shared by reward functions and DSPy inputs
# --------------------------------------------------------------------------------------


def strip_line_numbers(text: str) -> str:
    """
    Remove the ``'NNN: '` prefixes inserted by ``directive_extractor.py``.

    This is currently duplicated across the agents; moving it here lets each
    agent simply call ``core.strip_line_numbers``.
    """

    if not text:
        return ""
    return "\n".join(LINE_NUMBER_PATTERN.sub("", line) for line in text.splitlines())


def clamp_text_window(text: str, window: int = DEFAULT_CONTEXT_WINDOW) -> str:
    """
    Clamp ``text`` to a head/tail preview of at most ``window`` characters.

    The clamps match the behavior already implemented by the agents: roughly
    half of the snippet comes from the beginning and half from the end, with an
    ellipsis between them.
    """

    text = (text or "").strip()
    if not text or len(text) <= window:
        return text
    head = text[: window // 2]
    tail = text[-window // 2 :]
    return f"{head}\n...\n{tail}"


def _resolve_anchor_keys(obj: dict[str, Any], keys: Sequence[str]) -> str:
    """Internal helper for ``extract_context_window``."""
    for key in keys:
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    metadata = obj.get("metadata")
    if isinstance(metadata, dict):
        for key in keys:
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def extract_context_window(
    obj: dict[str, Any],
    document_text: str,
    *,
    window: int = DEFAULT_CONTEXT_WINDOW,
    primary_keys: Sequence[str] = ("label", "title", "term"),
) -> str:
    """
    Build the tiny context window passed to DSPy signatures.

    Preference order:
        1. The ``content`` field (already near the directive) if present.
        2. A substring around a key (label/title/term) in the full document.
    """

    content = strip_line_numbers(obj.get("content") or "").strip()
    if content:
        return clamp_text_window(content, window=window)

    anchor = _resolve_anchor_keys(obj, primary_keys)
    if not anchor:
        return ""

    idx = document_text.find(anchor)
    if idx == -1:
        anchor = anchor[: min(len(anchor), 60)]
        if not anchor:
            return ""
        idx = document_text.find(anchor)
    if idx == -1:
        return ""

    start = max(0, idx - window)
    end = min(len(document_text), idx + len(anchor) + window)
    return document_text[start:end]


def directive_text_from_object(
    obj: dict[str, Any],
    *,
    default_directive: str = "theorem",
    include_title: bool = True,
) -> str:
    """
    Return a best-effort directive text for DSPy, preferring ``raw_directive``.

    When ``raw_directive`` is missing the helper synthesizes a directive with:
        - ``::{prf:<directive_type>}``
        - optional title
        - optional ``:label:`` line
        - stripped ``content`` body
        - closing ``:::`` fence
    """

    raw = obj.get("raw_directive")
    if isinstance(raw, str) and raw.strip():
        return strip_line_numbers(raw)

    directive_type = (obj.get("directive_type") or default_directive).strip() or default_directive
    title = (obj.get("title") or "").strip()
    label = obj.get("label") or (obj.get("metadata") or {}).get("label")
    label_line = ""
    if isinstance(label, str) and label.strip():
        label_line = f":label: {label.strip()}"

    header = f"::{{prf:{directive_type}}}"
    if include_title and title:
        header = f"{header} {title}"

    body = strip_line_numbers(obj.get("content") or "")
    parts: list[str] = [header.strip()]
    if label_line:
        parts.append(label_line)
    if body.strip():
        parts.append(body.strip())
    parts.append(":::")
    return "\n\n".join(parts)


# --------------------------------------------------------------------------------------
# Small scoring/helper predicates (reused in reward functions)
# --------------------------------------------------------------------------------------


def nonempty(value: str | None) -> bool:
    """Utility predicate used in reward functions."""
    return bool(value and value.strip())


def fence_free(latex: str | None) -> bool:
    """Return ``True`` when ``latex`` does not contain display-math fences."""
    return not LATEX_FENCE_PATTERN.search(latex or "")


def reasonable_text_len(
    text: str | None,
    *,
    min_words: int = 5,
    max_chars: int = 600,
) -> bool:
    """
    Heuristic used by reward functions for natural-language snippets.

    DSPy rewards often check that outputs are neither too short nor overly long.
    """

    if not nonempty(text):
        return False
    words = len(text.strip().split())
    return words >= min_words and len(text) <= max_chars


def has_duplicates(items: Sequence[Any] | None) -> bool:
    """Return ``True`` if ``items`` contains duplicates."""
    if not items:
        return False
    seen = set()
    for entry in items:
        if entry in seen:
            return True
        seen.add(entry)
    return False


# --------------------------------------------------------------------------------------
# Convenience IO helpers
# --------------------------------------------------------------------------------------


def write_outputs(path: Path, payload: Any) -> None:
    """
    Write ``payload`` to ``path`` as JSON (UTF-8, pretty-printed).

    This helper is not exported in ``__all__`` yet, but leaving it here centralizes
    the ``ensure_ascii`` and indentation choices.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(safe_json_dumps(payload), encoding="utf-8")


# --------------------------------------------------------------------------------------
# Generic run loop scaffolding
# --------------------------------------------------------------------------------------


DirectiveInputBuilder = Callable[[dict[str, Any], str, str], dict[str, Any]]


def default_input_builder(
    obj: dict[str, Any],
    directive_text: str,
    context_hints: str,
) -> dict[str, Any]:
    """
    Default DSPy input payload: ``directive_text`` plus optional ``context_hints``.
    """

    payload: dict[str, Any] = {"directive_text": directive_text}
    if context_hints:
        payload["context_hints"] = context_hints
    return payload


def run_directive_extraction_loop(
    *,
    paths: DirectiveAgentPaths,
    program_call: Callable[..., Any],
    assemble_output: Callable[[Any], Any],
    directive_type_fallback: str,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
    primary_context_keys: Sequence[str] = ("label", "title", "term"),
    input_builder: DirectiveInputBuilder | None = None,
    fallback_call: Callable[..., Any] | None = None,
    logger_obj: logging.Logger | None = None,
) -> list[Any]:
    """
    Shared implementation of the "load → iterate → store" pattern.

    Parameters
    ----------
    paths:
        Precomputed document/registry/extract paths (see ``DirectiveAgentPaths``).
    program_call:
        Callable invoked for each directive (typically a ``dspy.Refine`` program).
    assemble_output:
        Function that converts the DSPy result into the final JSON object stored
        in the extract file.
    directive_type_fallback:
        Used when regenerating a directive header (e.g., "theorem", "definition").
    context_window:
        Character window for ``context_hints`` when full ``content`` is missing.
    primary_context_keys:
        Keys inspected (label/title/term) when anchoring into the document text.
    input_builder:
        Optional callable producing the kwargs passed to ``program_call``.
    fallback_call:
        Optional callable invoked when ``program_call`` raises.
    logger_obj:
        Optional logger override; defaults to this module's logger.

    Returns
    -------
    list:
        Collected outputs prior to serialization.
    """

    logger_use = logger_obj or logger
    if not paths.registry_path.exists():
        logger_use.warning("No directive registry found at %s", paths.registry_path)
        return []

    document_text = paths.document_path.read_text(encoding="utf-8")
    paths.ensure_output_directory()

    builder = input_builder or default_input_builder
    outputs: list[Any] = []
    all_items = list(iter_registry_items(paths.registry_path))
    for idx, obj in tqdm(
        enumerate(all_items, start=1), desc="Processing directives", total=len(all_items)
    ):
        directive_text = directive_text_from_object(obj, default_directive=directive_type_fallback)
        context_hints = extract_context_window(
            obj,
            document_text,
            window=context_window,
            primary_keys=primary_context_keys,
        )
        call_kwargs = builder(obj, directive_text, context_hints)
        label = obj.get("label")

        try:
            res = program_call(**call_kwargs)
            logger_use.info("✓ Refine succeeded for item #%s (%s)", idx, label)
        except Exception as exc:
            if fallback_call is None:
                logger_use.exception("Directive processing failed for item #%s (%s)", idx, label)
                raise
            logger_use.warning(
                "Refine failed for item #%s (%s); falling back. Reason: %s",
                idx,
                label,
                exc,
            )
            res = fallback_call(**call_kwargs)

        outputs.append(assemble_output(res))

    write_outputs(paths.extract_path, outputs)
    logger_use.info("Wrote %s directives to %s", len(outputs), paths.extract_path)
    return outputs
