"""Markdown renderer for :class:`UnifiedProposition`."""

from __future__ import annotations

from mathster.preprocess_extraction.data_models import UnifiedProposition
from mathster.reports.report_theorem import unified_theorem_to_markdown

__all__ = ["unified_proposition_to_markdown"]


def unified_proposition_to_markdown(proposition: UnifiedProposition) -> str:
    """Return the Markdown report for ``proposition`` using theorem renderer."""

    return unified_theorem_to_markdown(proposition)

