"""Report generators for enriched entity types → Markdown.

This package provides small utilities to render enriched data files into
formatted Markdown suited for Jupyter Book (MyST) docs.

Available renderers:
- equation_report: EquationBox → Markdown
- parameter_report: ParameterBox → Markdown
- remark_report: RemarkBox → Markdown

Router helper:
    from fragile.mathster.reports import render_enriched_to_markdown
    md = render_enriched_to_markdown(data_dict)
"""

from __future__ import annotations

from typing import Any, Dict

from mathster.reports.report_axiom import unified_axiom_to_markdown as preprocess_axiom_to_markdown
from mathster.reports.report_definition import unified_definition_to_markdown as preprocess_definition_to_markdown
from mathster.reports.report_remark import unified_remark_to_markdown as preprocess_remark_to_markdown
from mathster.reports.report_theorem import unified_theorem_to_markdown as preprocess_theorem_to_markdown


def _missing_renderer(name: str):
    def _renderer(*_args, **_kwargs):
        raise NotImplementedError(f"Renderer '{name}' is not available in this build.")

    return _renderer


try:  # pragma: no cover - legacy optional dependency
    from mathster.reports.axiom_report import axiom_to_markdown
except ModuleNotFoundError:  # pragma: no cover - fallback
    axiom_to_markdown = preprocess_axiom_to_markdown

try:  # pragma: no cover
    from mathster.reports.definition_report import definition_to_markdown
except ModuleNotFoundError:  # pragma: no cover
    definition_to_markdown = preprocess_definition_to_markdown

try:  # pragma: no cover
    from mathster.reports.equation_report import equation_to_markdown
except ModuleNotFoundError:  # pragma: no cover
    equation_to_markdown = _missing_renderer("equation")

try:  # pragma: no cover
    from mathster.reports.object_report import object_to_markdown
except ModuleNotFoundError:  # pragma: no cover
    object_to_markdown = _missing_renderer("object")

try:  # pragma: no cover
    from mathster.reports.parameter_report import parameter_to_markdown
except ModuleNotFoundError:  # pragma: no cover
    parameter_to_markdown = _missing_renderer("parameter")

try:  # pragma: no cover
    from mathster.reports.relationship_report import relationship_to_markdown
except ModuleNotFoundError:  # pragma: no cover
    relationship_to_markdown = _missing_renderer("relationship")

try:  # pragma: no cover
    from mathster.reports.remark_report import remark_to_markdown
except ModuleNotFoundError:  # pragma: no cover
    remark_to_markdown = preprocess_remark_to_markdown

try:  # pragma: no cover
    from mathster.reports.theorem_report import theorem_to_markdown
except ModuleNotFoundError:  # pragma: no cover
    theorem_to_markdown = preprocess_theorem_to_markdown

pipeline_equation_to_markdown = _missing_renderer("pipeline_equation")
pipeline_parameter_to_markdown = _missing_renderer("pipeline_parameter")
pipeline_remark_to_markdown = _missing_renderer("pipeline_remark")


__all__ = [
    "axiom_to_markdown",
    "definition_to_markdown",
    "equation_to_markdown",
    "object_to_markdown",
    "parameter_to_markdown",
    "relationship_to_markdown",
    "remark_to_markdown",
    "render_enriched_to_markdown",
    "theorem_to_markdown",
]


def render_enriched_to_markdown(data: dict[str, Any]) -> str:
    """Render any enriched entity dict to Markdown by label/type hint.

    Heuristic routing by label prefix:
    - eq-*: EquationBox
    - param-*: ParameterBox
    - remark-*: RemarkBox
    """
    label = str(data.get("label", ""))
    # Enriched/pipeline routing by label prefix
    if label.startswith("eq-"):
        # Prefer enriched EquationBox keys; fallback to pipeline shape
        try:
            return equation_to_markdown(data)
        except Exception:
            return pipeline_equation_to_markdown(data)
    if label.startswith("param-"):
        # Try enriched ParameterBox first; fallback to pipeline schema reporter
        try:
            return parameter_to_markdown(data)
        except Exception:
            return pipeline_parameter_to_markdown(data)
    if label.startswith("remark-"):
        try:
            return remark_to_markdown(data)
        except Exception:
            return pipeline_remark_to_markdown(data)
    if label.startswith("obj-"):
        return object_to_markdown(data)
    if label.startswith("axiom-"):
        return axiom_to_markdown(data)
    if label.startswith(("thm-", "lem-", "prop-", "cor-")):
        return theorem_to_markdown(data)
    if label.startswith("rel-"):
        return relationship_to_markdown(data)
    if label.startswith("def-"):
        return definition_to_markdown(data)
    msg = (
        "Unsupported enriched entity: cannot infer type from label. "
        "Expected prefixes: 'eq-', 'param-', or 'remark-'."
    )
    raise ValueError(msg)
