"""
Tests for parsing.preprocessing directive extraction.

Tests automated directive structure extraction from Jupyter Book MyST markdown.
"""

import pytest

from mathster.directives import extract_directive_hints, split_into_sections
from mathster.directives.directive_parser import DirectiveHint, extract_jupyter_directives


class TestDirectiveExtraction:
    """Test directive extraction from MyST markdown."""

    def test_extract_simple_definition(self):
        """Test extracting a simple definition directive."""
        text = """:::{prf:definition} Test Definition
:label: def-test

This is the definition content.
:::"""

        hints = extract_jupyter_directives(text)

        assert len(hints) == 1
        assert hints[0].directive_type == "definition"
        assert hints[0].label == "def-test"
        assert hints[0].title == "Test Definition"

    def test_extract_with_metadata(self):
        """Test extracting directive with multiple metadata fields."""
        text = """:::{prf:theorem} Main Result
:label: thm-main
:class: important
:nonumber:

Theorem statement here.
:::"""

        hints = extract_jupyter_directives(text)

        assert len(hints) == 1
        assert hints[0].label == "thm-main"
        assert hints[0].metadata.get("class") == "important"
        assert "nonumber" in hints[0].metadata

    def test_extract_with_line_numbers(self):
        """Test extraction from numbered text."""
        text = """001: :::{prf:definition} Test
002: :label: def-test
003:
004: Content here.
005: :::"""

        hints = extract_jupyter_directives(text)

        assert len(hints) == 1
        assert hints[0].label == "def-test"
        assert hints[0].start_line == 1
        assert hints[0].end_line == 5

    def test_extract_multiple_directives(self):
        """Test extracting multiple directives."""
        text = """:::{prf:definition} First
:label: def-1

Content 1
:::

Some text between.

:::{prf:theorem} Second
:label: thm-2

Content 2
:::"""

        hints = extract_jupyter_directives(text)

        assert len(hints) == 2
        assert hints[0].label == "def-1"
        assert hints[1].label == "thm-2"

    def test_content_boundaries(self):
        """Test that content boundaries are correctly identified."""
        text = """:::{prf:definition} Test
:label: def-test
:class: note

First line of content.
Second line of content.
:::"""

        hints = extract_jupyter_directives(text)

        assert len(hints) == 1
        assert "First line" in hints[0].content
        assert "Second line" in hints[0].content
        # Content should not include metadata lines
        assert ":label:" not in hints[0].content


class TestDirectiveExtractorModule:
    """Test the preprocessing.directive_extractor module."""

    def test_extract_directive_hints(self):
        """Test extract_directive_hints function."""
        text = """:::{prf:theorem} Test Theorem
:label: thm-test

Statement here.
:::"""

        hints = extract_directive_hints(text, chapter_number=0)

        assert len(hints) == 1
        assert isinstance(hints[0], dict)
        assert hints[0]["directive_type"] == "theorem"
        assert hints[0]["label"] == "thm-test"

    def test_hint_to_dict_conversion(self):
        """Test DirectiveHint.to_dict() method."""
        text = """:::{prf:definition} Test
:label: def-test

Content.
:::"""

        directives = extract_jupyter_directives(text)
        hint = directives[0]
        hint_dict = hint.to_dict()

        assert isinstance(hint_dict, dict)
        assert hint_dict["directive_type"] == "definition"
        assert hint_dict["label"] == "def-test"
        assert "start_line" in hint_dict
        assert "metadata" in hint_dict


class TestSplitIntoSectionsGranularity:
    """Tests for configurable heading scope in split_into_sections."""

    def test_split_by_level_two(self):
        """Sections split at level-2 headings should include deeper subheadings."""
        text = """# Chapter 1

## Section 1
Intro text.
### Subsection A
More details.

## Section 2
Other text.
"""
        sections = split_into_sections(text, heading_scope="##")
        assert [section.title for section in sections] == ["Section 1", "Section 2"]
        assert "### Subsection A" in sections[0].content

    def test_split_by_level_three(self):
        """Level-3 splitting should treat level-2 headings as outer context only."""
        text = """# Chapter

## Major Section
### Detail A
Content A.
### Detail B
Content B.
"""
        sections = split_into_sections(text, heading_scope="###")
        assert [section.title for section in sections] == ["Detail A", "Detail B"]
        assert all("## Major Section" not in section.content for section in sections)

    def test_invalid_heading_scope(self):
        """Non-# heading spec should raise ValueError."""
        with pytest.raises(ValueError):
            split_into_sections("# Title", heading_scope="invalid")
