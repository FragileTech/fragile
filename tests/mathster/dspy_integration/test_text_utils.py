"""
Tests for dspy_integration.text_utils module.

Tests generic text processing utilities: splitting, numbering, label sanitization.
"""

import pytest

from mathster.dspy_integration.text_utils import (
    add_line_numbers,
    sanitize_label,
    split_markdown_by_chapters,
)


class TestLabelSanitization:
    """Test label sanitization functions."""

    def test_sanitize_basic_labels(self):
        """Test basic label sanitization."""
        assert sanitize_label("def-test") == "def-test"
        assert sanitize_label("DEF-TEST") == "def-test"
        assert sanitize_label("Def_Test") == "def-test"

    def test_sanitize_special_characters(self):
        """Test handling of special characters."""
        assert sanitize_label("param_Gamma") == "param-gamma"
        assert sanitize_label("thm:main") == "thm-main"

    def test_sanitize_section_headers(self):
        """Test section header sanitization."""
        assert sanitize_label("## 1. Introduction") == "section-1-introduction"
        assert sanitize_label("# Chapter 2") == "chapter-2"  # Single # doesn't get section- prefix


class TestLineNumbering:
    """Test line numbering utilities."""

    def test_add_line_numbers(self):
        """Test adding line numbers to text."""
        text = "Line 1\nLine 2\nLine 3"
        numbered = add_line_numbers(text)

        assert "1: Line 1" in numbered
        assert "2: Line 2" in numbered
        assert "3: Line 3" in numbered

    def test_add_line_numbers_with_offset(self):
        """Test line numbering with offset."""
        text = "Line 1\nLine 2"
        numbered = add_line_numbers(text, offset=10)

        assert "011:" in numbered or "11:" in numbered
        assert "012:" in numbered or "12:" in numbered


class TestMarkdownSplitting:
    """Test markdown chapter splitting."""

    def test_split_function_exists(self):
        """Test that split function is callable."""
        assert callable(split_markdown_by_chapters)
        # Note: Actual splitting requires file path, tested in integration tests
