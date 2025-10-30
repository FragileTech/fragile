"""Shared pytest fixtures for proofs tests."""

import pytest

from fragile.proofs.core.article_system import SourceLocation


@pytest.fixture
def sample_source() -> SourceLocation:
    """Sample source location with complete information."""
    return SourceLocation(
        file_path="docs/source/1_euclidean_gas/01_test_document.md",
        chapter="1_euclidean_gas",
        document_id="01_test_document",
        section="1",
        section_name="Test Section",
        directive_label="test-directive",
        line_range=(10, 25),
    )


@pytest.fixture
def minimal_source() -> SourceLocation:
    """Minimal source location (document-level only)."""
    return SourceLocation(
        file_path="docs/source/1_euclidean_gas/01_test_document.md",
        chapter="1_euclidean_gas",
        document_id="01_test_document",
        section=None,
        section_name=None,
        directive_label=None,
        line_range=None,
    )


@pytest.fixture
def source_with_line_range() -> SourceLocation:
    """Source location with line range but no section."""
    return SourceLocation(
        file_path="docs/source/1_euclidean_gas/01_test_document.md",
        chapter="1_euclidean_gas",
        document_id="01_test_document",
        section=None,
        section_name=None,
        directive_label=None,
        line_range=(50, 75),
    )


@pytest.fixture
def source_with_section() -> SourceLocation:
    """Source location with section but no line range."""
    return SourceLocation(
        file_path="docs/source/1_euclidean_gas/01_test_document.md",
        chapter="1_euclidean_gas",
        document_id="01_test_document",
        section="2",
        section_name="Another Section",
        directive_label=None,
        line_range=None,
    )
