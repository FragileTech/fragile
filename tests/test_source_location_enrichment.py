"""
Tests for Source Location Enrichment Tools.

Tests cover:
- Text finding in markdown
- Directive finding
- Entity text extraction
- Source location creation with fallbacks
- Full enrichment workflow

Run with: pytest tests/test_source_location_enrichment.py
"""

import json
from pathlib import Path
import tempfile

import pytest

from fragile.proofs.core.article_system import SourceLocation
from fragile.proofs.tools.line_finder import (
    extract_lines,
    find_directive_lines,
    find_equation_lines,
    find_section_lines,
    find_text_in_markdown,
    validate_line_range,
)
from fragile.proofs.tools.source_location_enricher import (
    extract_directive_label_from_entity,
    extract_search_text_from_entity,
    find_entity_location,
)
from fragile.proofs.utils.source_helpers import SourceLocationBuilder


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_markdown():
    """Sample markdown content for testing."""
    return """# Introduction

This is the introduction section.

## Section 2: Main Results

:::{prf:theorem} thm-main
:label: thm-main

Let $v > 0$ be a velocity parameter. Assume the following conditions hold:
1. The domain is bounded
2. The potential is smooth

Then the convergence rate is exponential.
:::

The proof follows from Lemma 2.1.

:::{prf:lemma} lem-helper
:label: lem-helper

Under the conditions of Theorem 2.1, we have an explicit bound.
:::

Some display equation:

$$
\\mathbb{E}[V] = 0
$$

## Section 3: Extensions

More content here.
"""


@pytest.fixture
def sample_theorem_entity():
    """Sample raw theorem entity."""
    return {
        "temp_id": "raw-thm-1",
        "label_text": "thm-main",
        "statement_type": "theorem",
        "full_statement_text": "Let $v > 0$ be a velocity parameter. Assume the following conditions hold:\n1. The domain is bounded\n2. The potential is smooth\n\nThen the convergence rate is exponential.",
        "source_section": "§2",
    }


@pytest.fixture
def sample_definition_entity():
    """Sample raw definition entity."""
    return {
        "temp_id": "raw-def-1",
        "term_being_defined": "walker",
        "full_text": "A walker is a particle in the swarm that explores the state space.",
        "source_section": "§1",
    }


# =============================================================================
# LINE FINDER TESTS
# =============================================================================


def test_find_text_in_markdown(sample_markdown):
    """Test finding text snippets in markdown."""
    # Find theorem statement
    line_range = find_text_in_markdown(sample_markdown, "Let $v > 0$ be a velocity parameter")
    assert line_range is not None
    start, end = line_range
    assert start > 0
    assert end >= start

    # Verify extracted text matches
    extracted = extract_lines(sample_markdown, line_range)
    assert "Let $v > 0$" in extracted

    # Text not present
    line_range = find_text_in_markdown(sample_markdown, "This text does not exist")
    assert line_range is None


def test_find_directive_lines(sample_markdown):
    """Test finding Jupyter Book directives."""
    # Find theorem directive
    line_range = find_directive_lines(sample_markdown, "thm-main")
    assert line_range is not None
    start, end = line_range
    assert start > 0
    assert end >= start

    # Verify directive content
    extracted = extract_lines(sample_markdown, line_range)
    assert "prf:theorem" in extracted
    assert "thm-main" in extracted

    # Find lemma directive
    line_range = find_directive_lines(sample_markdown, "lem-helper", directive_type="lemma")
    assert line_range is not None

    # Directive not present
    line_range = find_directive_lines(sample_markdown, "thm-nonexistent")
    assert line_range is None


def test_find_equation_lines(sample_markdown):
    """Test finding LaTeX equations."""
    line_range = find_equation_lines(sample_markdown, "\\mathbb{E}[V] = 0")
    assert line_range is not None
    start, end = line_range
    assert start > 0
    assert end >= start

    # Verify equation content
    extracted = extract_lines(sample_markdown, line_range)
    assert "\\mathbb{E}[V]" in extracted or "mathbb{E}[V]" in extracted


def test_find_section_lines(sample_markdown):
    """Test finding sections by heading."""
    # Find section
    line_range = find_section_lines(sample_markdown, "Main Results")
    assert line_range is not None
    start, end = line_range
    assert start > 0
    assert end >= start

    # Verify section content
    extracted = extract_lines(sample_markdown, line_range)
    assert "Section 2" in extracted or "Main Results" in extracted

    # Section with exact match
    line_range = find_section_lines(sample_markdown, "Section 3: Extensions", exact_match=True)
    assert line_range is not None


def test_validate_line_range():
    """Test line range validation."""
    # Valid range
    assert validate_line_range((1, 10), 20) is True
    assert validate_line_range((5, 5), 20) is True  # Single line

    # Invalid ranges
    assert validate_line_range((0, 10), 20) is False  # Start < 1
    assert validate_line_range((10, 5), 20) is False  # Start > end
    assert validate_line_range((15, 25), 20) is False  # End > max_lines
    assert validate_line_range((-1, 10), 20) is False  # Negative


# =============================================================================
# ENTITY TEXT EXTRACTION TESTS
# =============================================================================


def test_extract_search_text_from_theorem(sample_theorem_entity):
    """Test extracting search text from theorem entity."""
    search_text = extract_search_text_from_entity(sample_theorem_entity)
    assert search_text is not None
    assert "Let $v > 0$" in search_text
    assert len(search_text) <= 200  # Should be truncated


def test_extract_search_text_from_definition(sample_definition_entity):
    """Test extracting search text from definition entity."""
    search_text = extract_search_text_from_entity(sample_definition_entity)
    assert search_text is not None
    assert "walker" in search_text
    assert "particle" in search_text


def test_extract_directive_label(sample_theorem_entity):
    """Test extracting directive label from entity."""
    label = extract_directive_label_from_entity(sample_theorem_entity)
    assert label == "thm-main"

    # Entity without label
    entity_no_label = {"temp_id": "raw-def-1"}
    label = extract_directive_label_from_entity(entity_no_label)
    assert label is None


# =============================================================================
# SOURCE LOCATION BUILDER TESTS
# =============================================================================


def test_source_location_builder_from_markdown_location():
    """Test creating SourceLocation from line range."""
    loc = SourceLocationBuilder.from_markdown_location(
        document_id="03_cloning",
        file_path="docs/source/1_euclidean_gas/03_cloning.md",
        start_line=142,
        end_line=158,
        section="§3.2",
    )

    assert loc.document_id == "03_cloning"
    assert loc.line_range == (142, 158)
    assert loc.section == "§3.2"
    assert loc.get_display_location() == "03_cloning (line 142-158)"


def test_source_location_builder_from_jupyter_directive():
    """Test creating SourceLocation from directive label."""
    loc = SourceLocationBuilder.from_jupyter_directive(
        document_id="03_cloning",
        file_path="docs/source/1_euclidean_gas/03_cloning.md",
        directive_label="thm-keystone",
        section="§3.2",
    )

    assert loc.document_id == "03_cloning"
    assert loc.directive_label == "thm-keystone"
    assert loc.url_fragment == "#thm-keystone"
    assert loc.section == "§3.2"


def test_source_location_builder_from_section():
    """Test creating SourceLocation from section only."""
    loc = SourceLocationBuilder.from_section(
        document_id="03_cloning",
        file_path="docs/source/1_euclidean_gas/03_cloning.md",
        section="§3.2",
    )

    assert loc.document_id == "03_cloning"
    assert loc.section == "§3.2"
    assert loc.line_range is None
    assert loc.directive_label is None


def test_source_location_builder_minimal():
    """Test creating minimal SourceLocation."""
    loc = SourceLocationBuilder.minimal(
        document_id="03_cloning", file_path="docs/source/1_euclidean_gas/03_cloning.md"
    )

    assert loc.document_id == "03_cloning"
    assert loc.file_path == "docs/source/1_euclidean_gas/03_cloning.md"
    assert loc.section is None
    assert loc.line_range is None


def test_source_location_builder_from_raw_entity(sample_theorem_entity):
    """Test creating SourceLocation from raw entity with fallback."""
    # With line range (most precise)
    loc = SourceLocationBuilder.from_raw_entity(
        entity_data=sample_theorem_entity,
        document_id="02_euclidean_gas",
        file_path="docs/source/1_euclidean_gas/02_euclidean_gas.md",
        line_range=(142, 158),
    )
    assert loc.line_range == (142, 158)
    assert loc.section == "§2"

    # Without line range (directive fallback)
    loc = SourceLocationBuilder.from_raw_entity(
        entity_data=sample_theorem_entity,
        document_id="02_euclidean_gas",
        file_path="docs/source/1_euclidean_gas/02_euclidean_gas.md",
    )
    assert loc.directive_label == "thm-main"
    assert loc.section == "§2"

    # Without line range or directive (section fallback)
    entity_no_label = {"source_section": "§3"}
    loc = SourceLocationBuilder.from_raw_entity(
        entity_data=entity_no_label,
        document_id="02_euclidean_gas",
        file_path="docs/source/1_euclidean_gas/02_euclidean_gas.md",
    )
    assert loc.section == "§3"
    assert loc.directive_label is None
    assert loc.line_range is None


def test_source_location_builder_with_fallback():
    """Test with_fallback method."""
    # With all fields (should use line range)
    loc = SourceLocationBuilder.with_fallback(
        document_id="03_cloning",
        file_path="docs/source/1_euclidean_gas/03_cloning.md",
        line_range=(142, 158),
        directive_label="thm-keystone",
        section="§3.2",
    )
    assert loc.line_range == (142, 158)

    # Without line range (should use directive)
    loc = SourceLocationBuilder.with_fallback(
        document_id="03_cloning",
        file_path="docs/source/1_euclidean_gas/03_cloning.md",
        directive_label="thm-keystone",
        section="§3.2",
    )
    assert loc.directive_label == "thm-keystone"
    assert loc.line_range is None

    # Section only
    loc = SourceLocationBuilder.with_fallback(
        document_id="03_cloning",
        file_path="docs/source/1_euclidean_gas/03_cloning.md",
        section="§3.2",
    )
    assert loc.section == "§3.2"
    assert loc.directive_label is None

    # Minimal
    loc = SourceLocationBuilder.with_fallback(
        document_id="03_cloning", file_path="docs/source/1_euclidean_gas/03_cloning.md"
    )
    assert loc.section is None
    assert loc.directive_label is None


# =============================================================================
# FULL ENRICHMENT WORKFLOW TESTS
# =============================================================================


def test_find_entity_location(sample_markdown, sample_theorem_entity):
    """Test finding entity location with fallback strategies."""
    loc = find_entity_location(
        entity_data=sample_theorem_entity,
        markdown_content=sample_markdown,
        document_id="02_euclidean_gas",
        file_path="docs/source/1_euclidean_gas/02_euclidean_gas.md",
    )

    assert loc is not None
    assert loc.document_id == "02_euclidean_gas"

    # Should find by text matching or directive label
    assert loc.line_range is not None or loc.directive_label is not None


def test_enrich_single_entity_workflow(sample_markdown, sample_theorem_entity):
    """Test the complete enrichment workflow for a single entity."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create temp markdown file
        markdown_file = tmpdir / "test_doc.md"
        markdown_file.write_text(sample_markdown)

        # Create temp JSON file
        json_file = tmpdir / "test_entity.json"
        json_file.write_text(json.dumps(sample_theorem_entity))

        # Import enricher function
        from fragile.proofs.tools.source_location_enricher import enrich_single_entity

        # Run enrichment
        success = enrich_single_entity(
            entity_json_path=json_file,
            markdown_file=markdown_file,
            document_id="02_euclidean_gas",
        )

        assert success is True

        # Verify enriched JSON
        enriched_data = json.loads(json_file.read_text())
        assert "source_location" in enriched_data

        loc = enriched_data["source_location"]
        assert loc["document_id"] == "02_euclidean_gas"
        assert loc["file_path"] == str(markdown_file)

        # Should have either line_range or directive_label
        assert "line_range" in loc or "directive_label" in loc


# =============================================================================
# EDGE CASES AND ERROR HANDLING
# =============================================================================


def test_find_text_case_sensitivity():
    """Test case-sensitive vs case-insensitive search."""
    content = "The Quick Brown Fox\nJumps Over The Lazy Dog"

    # Case-insensitive (default)
    line_range = find_text_in_markdown(content, "quick brown fox", case_sensitive=False)
    assert line_range is not None

    # Case-sensitive
    line_range = find_text_in_markdown(content, "quick brown fox", case_sensitive=True)
    assert line_range is None  # Won't match due to case

    line_range = find_text_in_markdown(content, "Quick Brown Fox", case_sensitive=True)
    assert line_range is not None  # Exact match


def test_multiline_text_search():
    """Test finding multi-line text snippets."""
    content = "Line 1\nLine 2\nLine 3\nLine 4"

    line_range = find_text_in_markdown(content, "Line 2\nLine 3")
    assert line_range is not None
    start, end = line_range
    assert start == 2
    assert end == 3


def test_extract_lines_edge_cases():
    """Test extract_lines with edge cases."""
    content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"

    # Extract first line
    extracted = extract_lines(content, (1, 1))
    assert extracted == "Line 1"

    # Extract last line
    extracted = extract_lines(content, (5, 5))
    assert extracted == "Line 5"

    # Extract all lines
    extracted = extract_lines(content, (1, 5))
    assert "Line 1" in extracted
    assert "Line 5" in extracted

    # Out of bounds should raise
    with pytest.raises(ValueError):
        extract_lines(content, (1, 10))


def test_entity_without_searchable_text():
    """Test entity that lacks searchable text fields."""
    minimal_entity = {"temp_id": "raw-unknown-1"}

    search_text = extract_search_text_from_entity(minimal_entity)
    assert search_text is None


def test_invalid_document_id():
    """Test handling of invalid document_id."""
    loc = find_entity_location(
        entity_data={},
        markdown_content="content",
        document_id="",  # Invalid: empty
        file_path="test.md",
    )
    assert loc is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
