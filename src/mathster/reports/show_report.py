"""Helper to render preprocess entities by label into Markdown reports."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping

from mathster.preprocess_extraction.data_models import (
    Algorithm,
    UnifiedAxiom,
    UnifiedCorollary,
    UnifiedDefinition,
    UnifiedLemma,
    UnifiedProof,
    UnifiedProposition,
    UnifiedRemark,
    UnifiedTheorem,
)
from mathster.registry.search import get_preprocess_label
from mathster.reports.report_algorithm import unified_algorithm_to_markdown
from mathster.reports.report_axiom import unified_axiom_to_markdown
from mathster.reports.report_corollary import unified_corollary_to_markdown
from mathster.reports.report_definition import unified_definition_to_markdown
from mathster.reports.report_lemma import unified_lemma_to_markdown
from mathster.reports.report_parameter import parameter_entry_to_markdown
from mathster.reports.report_proof import unified_proof_to_markdown
from mathster.reports.report_proposition import unified_proposition_to_markdown
from mathster.reports.report_remark import unified_remark_to_markdown
from mathster.reports.report_theorem import unified_theorem_to_markdown


Renderer = Callable[[Mapping[str, object]], str]


def render_label_report(label: str, preprocess_dir: Path | str | None = None) -> str:
    """Return the Markdown report for ``label`` sourced from preprocess registry."""

    entity = get_preprocess_label(label, preprocess_dir=preprocess_dir)
    if entity is None:
        raise ValueError(f"Unknown preprocess label '{label}'.")

    entity_type = _infer_type(label, entity)
    renderer = _ENTITY_RENDERERS.get(entity_type)
    if renderer is None:
        raise ValueError(f"Unsupported entity type '{entity_type}' for label '{label}'.")

    return renderer(entity)


def _infer_type(label: str, entity: Mapping[str, object]) -> str:
    entity_type = str(entity.get("type") or "").strip().lower()
    if entity_type:
        return entity_type

    prefix_map = {
        "thm-": "theorem",
        "lem-": "lemma",
        "cor-": "corollary",
        "prop-": "proposition",
        "axiom-": "axiom",
        "def-": "definition",
        "alg-": "algorithm",
        "remark-": "remark",
        "proof-": "proof",
        "obj-": "parameter",
    }
    for prefix, inferred in prefix_map.items():
        if label.startswith(prefix):
            return inferred

    if "signature" in entity and "steps" in entity:
        return "algorithm"
    if "remark_type" in entity:
        return "remark"

    raise ValueError(f"Cannot infer entity type for label '{label}'.")


def _wrap_model(model_cls: type, render_fn: Callable[[object], str]) -> Renderer:
    def _renderer(payload: Mapping[str, object]) -> str:
        model = model_cls(**payload)  # type: ignore[arg-type]
        return render_fn(model)

    return _renderer


_ENTITY_RENDERERS: dict[str, Renderer] = {
    "theorem": _wrap_model(UnifiedTheorem, unified_theorem_to_markdown),
    "lemma": _wrap_model(UnifiedLemma, unified_lemma_to_markdown),
    "corollary": _wrap_model(UnifiedCorollary, unified_corollary_to_markdown),
    "proposition": _wrap_model(UnifiedProposition, unified_proposition_to_markdown),
    "definition": _wrap_model(UnifiedDefinition, unified_definition_to_markdown),
    "axiom": _wrap_model(UnifiedAxiom, unified_axiom_to_markdown),
    "algorithm": _wrap_model(Algorithm, unified_algorithm_to_markdown),
    "remark": _wrap_model(UnifiedRemark, unified_remark_to_markdown),
    "proof": _wrap_model(UnifiedProof, unified_proof_to_markdown),
    "parameter": parameter_entry_to_markdown,
}
