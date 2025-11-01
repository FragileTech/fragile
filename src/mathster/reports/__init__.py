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

from mathster.reports.axiom_report import axiom_to_markdown
from mathster.reports.definition_report import definition_to_markdown
from mathster.reports.equation_report import equation_to_markdown
from mathster.reports.object_report import object_to_markdown
from mathster.reports.parameter_report import parameter_to_markdown
from mathster.reports.pipeline_equation_report import pipeline_equation_to_markdown
from mathster.reports.pipeline_parameter_report import pipeline_parameter_to_markdown
from mathster.reports.pipeline_remark_report import pipeline_remark_to_markdown
from mathster.reports.relationship_report import relationship_to_markdown
from mathster.reports.remark_report import remark_to_markdown
from mathster.reports.theorem_report import theorem_to_markdown


__all__ = [
    "axiom_to_markdown",
    "definition_to_markdown",
    "equation_to_markdown",
    "object_to_markdown",
    "parameter_to_markdown",
    "pipeline_equation_to_markdown",
    "pipeline_parameter_to_markdown",
    "pipeline_remark_to_markdown",
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
