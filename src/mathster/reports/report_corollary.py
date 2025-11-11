"""Markdown renderer for :class:`UnifiedCorollary`."""

from __future__ import annotations

from mathster.preprocess_extraction.data_models import UnifiedCorollary
from mathster.reports.report_theorem import unified_theorem_to_markdown


__all__ = ["unified_corollary_to_markdown"]


def unified_corollary_to_markdown(corollary: UnifiedCorollary) -> str:
    """Return the Markdown report for ``corollary`` using theorem renderer."""

    return unified_theorem_to_markdown(corollary)
