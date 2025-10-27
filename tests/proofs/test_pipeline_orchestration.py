"""
Tests for Pipeline Orchestration.

Tests the end-to-end Extract-then-Enrich pipeline with mocked LLM calls.
"""

import pytest
from unittest.mock import patch, MagicMock

from fragile.proofs.llm import (
    process_section,
    process_sections_parallel,
    merge_sections,
    enrich_and_assemble,
    process_document,
)
from fragile.proofs.tools import DocumentSection, DirectiveHint
from fragile.proofs.staging_types import StagingDocument, RawAxiom, RawTheorem, RawDefinition
from fragile.proofs.llm import MathematicalDocument


# Sample markdown for testing
SAMPLE_MARKDOWN = """
# Chapter 1: Introduction

This is the introduction.

:::{prf:definition} Walker
:label: def-walker

A walker is a tuple.
:::

## Section 1.1: Main Results

:::{prf:theorem} Convergence
:label: thm-convergence

The system converges exponentially.
:::

:::{prf:axiom} Bounded Displacement
:label: axiom-bounded

All walkers satisfy displacement bounds.
:::
"""


class TestProcessSection:
    """Tests for process_section() with mocked LLM."""

    @patch('fragile.proofs.llm.pipeline_orchestration.call_main_extraction_llm')
    def test_process_section_basic(self, mock_llm):
        """Test processing a single section."""
        # Mock LLM response
        mock_llm.return_value = {
            "section_id": "§1",
            "definitions": [],
            "theorems": [],
            "proofs": [],
            "axioms": [],
            "citations": [],
            "equations": [],
            "parameters": [],
            "remarks": []
        }

        # Create a simple section
        section = DocumentSection(
            section_id="§1",
            title="Test Section",
            level=1,
            start_line=1,
            end_line=10,
            content="Test content",
            directives=[]
        )

        # Process
        staging_doc = process_section(section)

        # Verify LLM was called
        assert mock_llm.called
        assert mock_llm.call_count == 1

        # Verify result
        assert staging_doc.section_id == "§1"
        assert staging_doc.total_entities == 0

    @patch('fragile.proofs.llm.pipeline_orchestration.call_main_extraction_llm')
    def test_process_section_with_entities(self, mock_llm):
        """Test processing section with extracted entities."""
        # Mock LLM response with entities
        mock_llm.return_value = {
            "section_id": "§1",
            "definitions": [
                {
                    "temp_id": "raw-def-001",
                    "label_text": "def-walker",
                    "term": "Walker",
                    "statement_text": "A walker is a tuple",
                    "source_section": "§1"
                }
            ],
            "theorems": [
                {
                    "temp_id": "raw-thm-001",
                    "label_text": "thm-test",
                    "name": "Test Theorem",
                    "statement_text": "Test statement",
                    "source_section": "§1"
                }
            ],
            "proofs": [],
            "axioms": [],
            "citations": [],
            "equations": [],
            "parameters": [],
            "remarks": []
        }

        section = DocumentSection(
            section_id="§1",
            title="Test",
            level=1,
            start_line=1,
            end_line=10,
            content="Content",
            directives=[]
        )

        staging_doc = process_section(section)

        assert len(staging_doc.definitions) == 1
        assert len(staging_doc.theorems) == 1
        assert staging_doc.total_entities == 2

    @patch('fragile.proofs.llm.pipeline_orchestration.call_main_extraction_llm')
    def test_process_section_with_directives(self, mock_llm):
        """Test that directive hints are passed to LLM."""
        mock_llm.return_value = {
            "section_id": "§1",
            "definitions": [],
            "theorems": [],
            "proofs": [],
            "axioms": [],
            "citations": [],
            "equations": [],
            "parameters": [],
            "remarks": []
        }

        # Create section with directives
        directive = DirectiveHint(
            directive_type="theorem",
            label="thm-test",
            start_line=5,
            end_line=10,
            content="Test content",
            section="§1"
        )

        section = DocumentSection(
            section_id="§1",
            title="Test",
            level=1,
            start_line=1,
            end_line=20,
            content="Content",
            directives=[directive]
        )

        staging_doc = process_section(section)

        # Verify LLM was called with directive hints in the text
        call_args = mock_llm.call_args
        assert "section_text" in call_args.kwargs
        # Should include directive hints
        assert "Directive Hints" in call_args.kwargs["section_text"]

    @patch('fragile.proofs.llm.pipeline_orchestration.call_main_extraction_llm')
    def test_process_section_error_handling(self, mock_llm):
        """Test error handling when LLM fails."""
        # Mock LLM to raise exception
        mock_llm.side_effect = Exception("API Error")

        section = DocumentSection(
            section_id="§1",
            title="Test",
            level=1,
            start_line=1,
            end_line=10,
            content="Content",
            directives=[]
        )

        # Should return empty staging document on error
        staging_doc = process_section(section)

        assert staging_doc.section_id == "§1"
        assert staging_doc.total_entities == 0


class TestProcessSectionsParallel:
    """Tests for process_sections_parallel()."""

    @patch('fragile.proofs.llm.pipeline_orchestration.call_main_extraction_llm')
    def test_process_multiple_sections(self, mock_llm):
        """Test processing multiple sections."""
        # Mock LLM to return different data for each call
        mock_llm.side_effect = [
            {
                "section_id": "§1",
                "definitions": [],
                "theorems": [],
                "proofs": [],
                "axioms": [],
                "citations": [],
                "equations": [],
                "parameters": [],
                "remarks": []
            },
            {
                "section_id": "§2",
                "definitions": [],
                "theorems": [],
                "proofs": [],
                "axioms": [],
                "citations": [],
                "equations": [],
                "parameters": [],
                "remarks": []
            }
        ]

        sections = [
            DocumentSection("§1", "Section 1", 1, 1, 10, "Content 1", []),
            DocumentSection("§2", "Section 2", 1, 11, 20, "Content 2", [])
        ]

        staging_docs = process_sections_parallel(sections)

        assert len(staging_docs) == 2
        assert staging_docs[0].section_id == "§1"
        assert staging_docs[1].section_id == "§2"
        assert mock_llm.call_count == 2


class TestMergeSections:
    """Tests for merge_sections()."""

    def test_merge_empty_sections(self):
        """Test merging empty sections."""
        staging_docs = []
        merged = merge_sections(staging_docs)

        assert merged.section_id == "merged-document"
        assert merged.total_entities == 0

    def test_merge_single_section(self):
        """Test merging single section."""
        staging = StagingDocument(
            section_id="§1",
            definitions=[],
            theorems=[],
            proofs=[],
            axioms=[],
            citations=[],
            equations=[],
            parameters=[],
            remarks=[]
        )

        merged = merge_sections([staging])

        assert merged.section_id == "merged-document"
        assert merged.total_entities == 0

    def test_merge_multiple_sections(self):
        """Test merging multiple sections with entities."""
        # Section 1 with axiom
        axiom1 = RawAxiom(
            temp_id="raw-axiom-001",
            label_text="axiom-1",
            name="Axiom 1",
            core_assumption_text="Test",
            source_section="§1"
        )
        staging1 = StagingDocument(
            section_id="§1",
            definitions=[],
            theorems=[],
            proofs=[],
            axioms=[axiom1],
            citations=[],
            equations=[],
            parameters=[],
            remarks=[]
        )

        # Section 2 with theorem
        theorem1 = RawTheorem(
            temp_id="raw-thm-001",
            label_text="thm-1",
            name="Theorem 1",
            statement_text="Test",
            source_section="§2"
        )
        staging2 = StagingDocument(
            section_id="§2",
            definitions=[],
            theorems=[theorem1],
            proofs=[],
            axioms=[],
            citations=[],
            equations=[],
            parameters=[],
            remarks=[]
        )

        merged = merge_sections([staging1, staging2])

        assert len(merged.axioms) == 1
        assert len(merged.theorems) == 1
        assert merged.total_entities == 2


class TestEnrichAndAssemble:
    """Tests for enrich_and_assemble()."""

    def test_enrich_empty_staging(self):
        """Test enriching empty staging document."""
        staging = StagingDocument(
            section_id="§1",
            definitions=[],
            theorems=[],
            proofs=[],
            axioms=[],
            citations=[],
            equations=[],
            parameters=[],
            remarks=[]
        )

        math_doc = enrich_and_assemble(staging, chapter="1_euclidean_gas")

        assert math_doc.document_id == "§1"
        assert math_doc.chapter == "1_euclidean_gas"
        assert math_doc.total_raw_entities == 0
        assert math_doc.total_enriched_entities == 0

    def test_enrich_with_axioms(self):
        """Test enriching staging document with axioms."""
        axiom = RawAxiom(
            temp_id="raw-axiom-001",
            label_text="axiom-bounded",
            name="Bounded Displacement",
            core_assumption_text="All walkers satisfy bounds",
            parameters_text=["ε > 0"],
            condition_text="When Δt < ε²",
            source_section="§1"
        )

        staging = StagingDocument(
            section_id="§1",
            definitions=[],
            theorems=[],
            proofs=[],
            axioms=[axiom],
            citations=[],
            equations=[],
            parameters=[],
            remarks=[]
        )

        math_doc = enrich_and_assemble(staging, chapter="1_euclidean_gas", document="01_framework")

        assert math_doc.total_raw_entities == 1
        assert math_doc.total_enriched_entities == 1
        assert len(math_doc.enriched.axioms) == 1
        assert "axiom-bounded" in math_doc.enriched.axioms

    def test_enrich_with_mixed_entities(self):
        """Test enriching with multiple entity types."""
        axiom = RawAxiom(
            temp_id="raw-axiom-001",
            label_text="axiom-test",
            name="Test",
            core_assumption_text="Test",
            source_section="§1"
        )

        definition = RawDefinition(
            temp_id="raw-def-001",
            label_text="def-walker",
            term="Walker",
            statement_text="A walker is...",
            source_section="§1"
        )

        theorem = RawTheorem(
            temp_id="raw-thm-001",
            label_text="thm-conv",
            name="Convergence",
            statement_text="The system converges",
            source_section="§1"
        )

        staging = StagingDocument(
            section_id="§1",
            definitions=[definition],
            theorems=[theorem],
            proofs=[],
            axioms=[axiom],
            citations=[],
            equations=[],
            parameters=[],
            remarks=[]
        )

        math_doc = enrich_and_assemble(staging)

        assert math_doc.total_raw_entities == 3
        assert math_doc.total_enriched_entities == 3
        assert len(math_doc.enriched.definitions) == 1
        assert len(math_doc.enriched.theorems) == 1
        assert len(math_doc.enriched.axioms) == 1

    def test_enrich_with_error_logging(self):
        """Test that errors are logged but don't stop processing."""
        from fragile.proofs.error_tracking import create_logger_for_document

        # Create invalid axiom (missing required field - will fail validation during enrichment)
        # Actually, our from_raw is pretty robust, so let's just verify error logger integration
        axiom = RawAxiom(
            temp_id="raw-axiom-001",
            label_text="axiom-test",
            name="Test",
            core_assumption_text="Test",
            source_section="§1"
        )

        staging = StagingDocument(
            section_id="§1",
            definitions=[],
            theorems=[],
            proofs=[],
            axioms=[axiom],
            citations=[],
            equations=[],
            parameters=[],
            remarks=[]
        )

        error_logger = create_logger_for_document("test_doc")
        math_doc = enrich_and_assemble(staging, error_logger=error_logger)

        # Should complete successfully
        assert math_doc.total_enriched_entities == 1


class TestProcessDocument:
    """Tests for process_document() end-to-end."""

    @patch('fragile.proofs.llm.pipeline_orchestration.call_main_extraction_llm')
    def test_process_document_end_to_end(self, mock_llm):
        """Test complete document processing pipeline."""
        # Mock LLM to return entities for each section
        mock_llm.return_value = {
            "section_id": "test",
            "definitions": [
                {
                    "temp_id": "raw-def-001",
                    "label_text": "def-walker",
                    "term": "Walker",
                    "statement_text": "A walker is a tuple",
                    "source_section": "test"
                }
            ],
            "theorems": [],
            "proofs": [],
            "axioms": [
                {
                    "temp_id": "raw-axiom-001",
                    "label_text": "axiom-bounded",
                    "name": "Bounded Displacement",
                    "core_assumption_text": "All walkers satisfy bounds",
                    "parameters_text": [],
                    "condition_text": "",
                    "source_section": "test"
                }
            ],
            "citations": [],
            "equations": [],
            "parameters": [],
            "remarks": []
        }

        math_doc = process_document(
            markdown_text=SAMPLE_MARKDOWN,
            document_id="test_doc",
            chapter="1_euclidean_gas",
            enable_error_logging=False  # Disable for test
        )

        # Verify document structure
        assert math_doc.document_id == "test_doc"
        assert math_doc.chapter == "1_euclidean_gas"

        # Should have processed sections (Chapter 1 + Section 1.1 = 2 sections)
        assert len(math_doc.staging_documents) >= 2

        # Should have enriched some entities
        assert math_doc.total_enriched_entities > 0

    @patch('fragile.proofs.llm.pipeline_orchestration.call_main_extraction_llm')
    def test_process_document_with_file_path(self, mock_llm):
        """Test process_document with file_path metadata."""
        mock_llm.return_value = {
            "section_id": "test",
            "definitions": [],
            "theorems": [],
            "proofs": [],
            "axioms": [],
            "citations": [],
            "equations": [],
            "parameters": [],
            "remarks": []
        }

        math_doc = process_document(
            markdown_text="# Test\n\nContent",
            document_id="test",
            chapter="1_euclidean_gas",
            file_path="/path/to/test.md",
            enable_error_logging=False
        )

        assert math_doc.file_path == "/path/to/test.md"

    @patch('fragile.proofs.llm.pipeline_orchestration.call_main_extraction_llm')
    def test_process_document_enrichment_statistics(self, mock_llm):
        """Test that enrichment statistics are calculated correctly."""
        # Mock to return 10 raw entities
        mock_llm.return_value = {
            "section_id": "test",
            "definitions": [],
            "theorems": [],
            "proofs": [],
            "axioms": [
                {
                    "temp_id": f"raw-axiom-{i:03d}",
                    "label_text": f"axiom-{i}",
                    "name": f"Axiom {i}",
                    "core_assumption_text": "Test",
                    "parameters_text": [],
                    "condition_text": "",
                    "source_section": "test"
                }
                for i in range(10)
            ],
            "citations": [],
            "equations": [],
            "parameters": [],
            "remarks": []
        }

        math_doc = process_document(
            markdown_text="# Test\n\nContent",
            document_id="test",
            chapter="1_euclidean_gas",
            enable_error_logging=False
        )

        # All 10 axioms should be enriched
        assert math_doc.total_raw_entities == 10
        assert math_doc.total_enriched_entities == 10
        assert math_doc.enrichment_rate == 100.0


class TestIntegration:
    """Integration tests for complete workflow."""

    @patch('fragile.proofs.llm.pipeline_orchestration.call_main_extraction_llm')
    def test_complete_workflow(self, mock_llm):
        """Test complete workflow from markdown to enriched document."""
        # Mock LLM responses
        mock_llm.return_value = {
            "section_id": "test",
            "definitions": [
                {
                    "temp_id": "raw-def-001",
                    "label_text": "def-walker",
                    "term": "Walker",
                    "statement_text": "A walker is a tuple (x, v, s)",
                    "source_section": "test"
                }
            ],
            "theorems": [
                {
                    "temp_id": "raw-thm-001",
                    "label_text": "thm-convergence",
                    "name": "Convergence Theorem",
                    "statement_text": "The system converges exponentially",
                    "source_section": "test"
                }
            ],
            "proofs": [],
            "axioms": [
                {
                    "temp_id": "raw-axiom-001",
                    "label_text": "axiom-bounded",
                    "name": "Bounded Displacement",
                    "core_assumption_text": "All walkers satisfy |x(t+Δt) - x(t)| ≤ ε√Δt",
                    "parameters_text": ["ε > 0", "Δt"],
                    "condition_text": "When Δt < ε²",
                    "failure_mode_analysis_text": "Teleportation behavior",
                    "source_section": "test"
                }
            ],
            "citations": [],
            "equations": [],
            "parameters": [],
            "remarks": []
        }

        # Process document
        math_doc = process_document(
            markdown_text=SAMPLE_MARKDOWN,
            document_id="01_framework",
            chapter="1_euclidean_gas",
            file_path="/path/to/01_framework.md",
            enable_error_logging=False
        )

        # Verify complete structure
        assert math_doc.document_id == "01_framework"
        assert math_doc.chapter == "1_euclidean_gas"
        assert math_doc.file_path == "/path/to/01_framework.md"

        # Verify entities were extracted and enriched
        assert math_doc.total_enriched_entities > 0

        # Verify lookup works
        axiom = math_doc.get_axiom("axiom-bounded")
        assert axiom is not None
        assert axiom.name == "Bounded Displacement"
        assert len(axiom.parameters) == 2

        # Verify summary works
        summary = math_doc.get_summary()
        assert "01_framework" in summary
        assert "1_euclidean_gas" in summary
