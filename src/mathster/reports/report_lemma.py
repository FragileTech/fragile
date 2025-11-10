"""Markdown renderer for :class:`UnifiedLemma`."""

from __future__ import annotations

from mathster.preprocess_extraction.data_models import UnifiedLemma
from mathster.reports.report_theorem import unified_theorem_to_markdown

__all__ = ["unified_lemma_to_markdown"]


def unified_lemma_to_markdown(lemma: UnifiedLemma) -> str:
    """Return the Markdown report for ``lemma`` using theorem renderer."""

    return unified_theorem_to_markdown(lemma)

