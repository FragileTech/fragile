"""
Tests for Directive Parser.

Tests the hybrid parsing approach: directive extraction and section splitting.
"""

import pytest

from fragile.proofs.tools import (
    DirectiveHint,
    DocumentSection,
    extract_jupyter_directives,
    format_directive_hints_for_llm,
    generate_section_id,
    get_directive_summary,
    split_into_sections,
)


class TestExtractJupyterDirectives:
    """Tests for extract_jupyter_directives() function."""

    def test_extract_single_theorem(self):
        """Test extracting a single theorem directive."""
        markdown = """
:::{prf:theorem} Main Result
:label: thm-main

Under the stated assumptions, the system converges.
:::
"""
        directives = extract_jupyter_directives(markdown, section_id="§1")

        assert len(directives) == 1
        assert directives[0].directive_type == "theorem"
        assert directives[0].label == "thm-main"
        assert "converges" in directives[0].content
        assert directives[0].section == "§1"

    def test_extract_multiple_directives(self):
        """Test extracting multiple directives."""
        markdown = """
:::{prf:definition} Walker
:label: def-walker

A walker is a tuple.
:::

Some text in between.

:::{prf:theorem} Convergence
:label: thm-conv

The system converges.
:::

:::{prf:proof}
:label: proof-conv

We proceed by induction.
:::
"""
        directives = extract_jupyter_directives(markdown)

        assert len(directives) == 3
        assert directives[0].directive_type == "definition"
        assert directives[0].label == "def-walker"
        assert directives[1].directive_type == "theorem"
        assert directives[1].label == "thm-conv"
        assert directives[2].directive_type == "proof"
        assert directives[2].label == "proof-conv"

    def test_extract_directive_without_label(self):
        """Test handling directive without label."""
        markdown = """
:::{prf:remark}

This is a remark without a label.
:::
"""
        directives = extract_jupyter_directives(markdown)

        assert len(directives) == 1
        assert directives[0].directive_type == "remark"
        # Should have auto-generated label
        assert "unlabeled" in directives[0].label

    def test_extract_directive_line_numbers(self):
        """Test that line numbers are correct (1-indexed)."""
        markdown = """Line 1
Line 2
:::{prf:theorem} Test
:label: thm-test

Content line.
:::
Line 8
"""
        directives = extract_jupyter_directives(markdown)

        assert len(directives) == 1
        # Directive starts at line 3 (1-indexed)
        assert directives[0].start_line == 3
        # Directive ends at line 7 (1-indexed)
        assert directives[0].end_line == 7

    def test_extract_nested_content(self):
        """Test extracting content with nested formatting."""
        markdown = """
:::{prf:definition} Complex
:label: def-complex

This definition contains:
- **Bold text**
- *Italic text*
- $\\LaTeX$ math: $x \\in \\mathbb{R}$

And a code block:
```python
x = 1
```
:::
"""
        directives = extract_jupyter_directives(markdown)

        assert len(directives) == 1
        content = directives[0].content
        assert "**Bold text**" in content
        assert "*Italic text*" in content
        assert "$\\LaTeX$" in content
        assert "```python" in content

    def test_extract_multiple_field_lines(self):
        """Test directives with multiple field lines."""
        markdown = """
:::{prf:theorem} Test Theorem
:label: thm-test
:nonumber:

Content here.
:::
"""
        directives = extract_jupyter_directives(markdown)

        assert len(directives) == 1
        assert directives[0].label == "thm-test"

    def test_extract_empty_document(self):
        """Test empty document returns empty list."""
        directives = extract_jupyter_directives("")
        assert directives == []

    def test_extract_no_directives(self):
        """Test document without directives."""
        markdown = """
# Regular Heading

Some regular text without any directives.

- List item 1
- List item 2
"""
        directives = extract_jupyter_directives(markdown)
        assert directives == []


class TestSplitIntoSections:
    """Tests for split_into_sections() function."""

    def test_split_single_section(self):
        """Test document with single heading."""
        markdown = """
# Chapter 1: Introduction

This is the introduction content.
"""
        sections = split_into_sections(markdown)

        assert len(sections) == 1
        assert sections[0].title == "Chapter 1: Introduction"
        assert sections[0].level == 1
        assert "introduction content" in sections[0].content

    def test_split_multiple_sections(self):
        """Test document with multiple headings."""
        markdown = """
# Chapter 1

Introduction here.

## Section 1.1

First subsection.

## Section 1.2

Second subsection.

# Chapter 2

New chapter.
"""
        sections = split_into_sections(markdown)

        assert len(sections) == 4
        assert sections[0].title == "Chapter 1"
        assert sections[0].level == 1
        assert sections[1].title == "Section 1.1"
        assert sections[1].level == 2
        assert sections[2].title == "Section 1.2"
        assert sections[3].title == "Chapter 2"

    def test_split_preserves_directives(self):
        """Test that directives are extracted for each section."""
        markdown = """
# Section 1

:::{prf:theorem} Test
:label: thm-test

Content.
:::

## Section 1.1

:::{prf:definition} Def
:label: def-test

Definition.
:::
"""
        sections = split_into_sections(markdown)

        assert len(sections) == 2
        assert len(sections[0].directives) == 1
        assert sections[0].directives[0].directive_type == "theorem"
        assert len(sections[1].directives) == 1
        assert sections[1].directives[0].directive_type == "definition"

    def test_split_no_headings(self):
        """Test document without headings."""
        markdown = """
This is content without any headings.

Just regular paragraphs.
"""
        sections = split_into_sections(markdown)

        # Should create single "main" section
        assert len(sections) == 1
        assert sections[0].section_id == "main"
        assert sections[0].title == "Main Document"

    def test_split_line_numbers(self):
        """Test that line numbers are correct."""
        markdown = """Line 1
# First Heading
Line 3
Line 4
## Second Heading
Line 6
"""
        sections = split_into_sections(markdown)

        assert len(sections) == 2
        # First section starts at line 2 (heading line)
        assert sections[0].start_line == 2
        # Second section starts at line 5 (heading line)
        assert sections[1].start_line == 5


class TestGenerateSectionId:
    """Tests for generate_section_id() function."""

    def test_generate_id_with_number(self):
        """Test ID generation with section number."""
        assert generate_section_id("Chapter 1: Introduction", 0) == "§1-introduction"
        assert generate_section_id("Section 2.1: Main Results", 10) == "§2.1-main-results"
        assert generate_section_id("3.4.2 Advanced Topics", 20) == "§3.4.2-advanced-topics"

    def test_generate_id_without_number(self):
        """Test ID generation without section number."""
        assert generate_section_id("Introduction", 0) == "introduction"
        assert generate_section_id("Main Results", 10) == "main-results"
        assert (
            generate_section_id("Conclusion and Future Work", 20) == "conclusion-and-future-work"
        )

    def test_generate_id_special_characters(self):
        """Test handling of special characters."""
        assert generate_section_id("Section 2.1: C++ Programming", 0) == "§2.1-c-programming"
        assert generate_section_id("Intro: (Part 1)", 0) == "intro-part-1"

    def test_generate_id_case_insensitive(self):
        """Test that IDs are lowercase."""
        assert generate_section_id("CHAPTER 1", 0) == "§1-"
        assert generate_section_id("Introduction", 0) == "introduction"


class TestGetDirectiveSummary:
    """Tests for get_directive_summary() function."""

    def test_summary_single_type(self):
        """Test summary with single directive type."""
        directives = [
            DirectiveHint("theorem", "thm-1", 1, 10, "...", "§1"),
            DirectiveHint("theorem", "thm-2", 11, 20, "...", "§1"),
            DirectiveHint("theorem", "thm-3", 21, 30, "...", "§1"),
        ]

        summary = get_directive_summary(directives)
        assert summary == {"theorem": 3}

    def test_summary_multiple_types(self):
        """Test summary with multiple directive types."""
        directives = [
            DirectiveHint("theorem", "thm-1", 1, 10, "...", "§1"),
            DirectiveHint("definition", "def-1", 11, 20, "...", "§1"),
            DirectiveHint("theorem", "thm-2", 21, 30, "...", "§1"),
            DirectiveHint("proof", "proof-1", 31, 40, "...", "§1"),
        ]

        summary = get_directive_summary(directives)
        assert summary == {"theorem": 2, "definition": 1, "proof": 1}

    def test_summary_empty_list(self):
        """Test summary with empty list."""
        summary = get_directive_summary([])
        assert summary == {}


class TestFormatDirectiveHintsForLLM:
    """Tests for format_directive_hints_for_llm() function."""

    def test_format_single_directive(self):
        """Test formatting single directive."""
        directives = [DirectiveHint("theorem", "thm-test", 10, 25, "...", "§2")]

        formatted = format_directive_hints_for_llm(directives)

        assert "Directive Hints:" in formatted
        assert "[theorem]" in formatted
        assert "thm-test" in formatted
        assert "lines 10-25" in formatted

    def test_format_multiple_directives(self):
        """Test formatting multiple directives."""
        directives = [
            DirectiveHint("theorem", "thm-1", 10, 20, "...", "§1"),
            DirectiveHint("definition", "def-1", 30, 40, "...", "§1"),
            DirectiveHint("proof", "proof-1", 50, 60, "...", "§1"),
        ]

        formatted = format_directive_hints_for_llm(directives)

        assert "[theorem]" in formatted
        assert "[definition]" in formatted
        assert "[proof]" in formatted
        assert "thm-1" in formatted
        assert "def-1" in formatted
        assert "proof-1" in formatted

    def test_format_empty_list(self):
        """Test formatting empty list."""
        formatted = format_directive_hints_for_llm([])
        assert "No directive hints available" in formatted


class TestDirectiveHintDataclass:
    """Tests for DirectiveHint dataclass."""

    def test_directive_hint_creation(self):
        """Test creating DirectiveHint."""
        hint = DirectiveHint(
            directive_type="theorem",
            label="thm-test",
            start_line=10,
            end_line=25,
            content="Test content",
            section="§1",
        )

        assert hint.directive_type == "theorem"
        assert hint.label == "thm-test"
        assert hint.start_line == 10
        assert hint.end_line == 25
        assert hint.content == "Test content"
        assert hint.section == "§1"


class TestDocumentSectionDataclass:
    """Tests for DocumentSection dataclass."""

    def test_document_section_creation(self):
        """Test creating DocumentSection."""
        directive = DirectiveHint("theorem", "thm-1", 5, 10, "...", "§1")

        section = DocumentSection(
            section_id="§1-intro",
            title="Introduction",
            level=1,
            start_line=1,
            end_line=50,
            content="Section content here",
            directives=[directive],
        )

        assert section.section_id == "§1-intro"
        assert section.title == "Introduction"
        assert section.level == 1
        assert section.start_line == 1
        assert section.end_line == 50
        assert len(section.directives) == 1
        assert section.directives[0].directive_type == "theorem"
